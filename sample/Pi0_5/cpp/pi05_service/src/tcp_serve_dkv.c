//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
/* tcp_serve.c — pi0.5 SE7 常驻 TCP 推理服务: JPG 传输 + JPU 硬解(jpeg_bm) + 完整推理链.
 *
 * 架构: 本进程 = TCP server + JPEG 解码 + 前后处理;
 *       子进程 serve_infer2 = siglip + 4 段 denoise bmodel (显式 pipe 双向).
 * 协议(小端, 每请求):
 *   请求: [1B tidx][1B steps][8B seed][4B len0][4B len1][noise 320xf32][jpg0][jpg1]  (头 18B + 1280B noise)
 *   响应: [1B ok=1][1280B traj f32 10x32]  或 [1B ok=0]
 * tidx 选语言/amask/cossin 资产(prompt200_tXX.npy / amask_978_tXX.npy / cossin_978_tXX.npz).
 * usage: tcp_serve <devid> <workdir> <port>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <arpa/inet.h>
#include <signal.h>
#include <time.h>
#include <math.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>

#define IMG_N 224
#define VIS_TOK 512              /* H5: 只保留 2 路真实图像(第 3 路黑图 token 已删) */
#define PL 536   /* H5: 968 -> 536(删 436 死 token; 留槽给存活 533 的 t04) */
#define L_SEQ 546   /* H5: PL+AH */
#define AH 10
#define TRAJ_B 1280

static int g_srv = -1;

/* ---------- JPEG 解码 (JPU) ---------- */
typedef struct {
    AVCodecContext* ctx;
    AVPacket* pkt;
    AVFrame* frame;
    struct SwsContext* sws;
    uint8_t* rgb;
} jpgdec_t;

static int jpgdec_init(jpgdec_t* d){
    memset(d,0,sizeof(*d));
    const AVCodec* c = avcodec_find_decoder_by_name("jpeg_bm");
    if(!c){fprintf(stderr,"jpeg_bm decoder not found\n");return -1;}
    d->ctx = avcodec_alloc_context3(c);
    if(avcodec_open2(d->ctx,c,NULL)<0){fprintf(stderr,"jpeg_bm open fail\n");return -1;}
    d->pkt = av_packet_alloc();
    d->frame = av_frame_alloc();
    d->rgb = malloc(IMG_N*IMG_N*3);
    return 0;
}

static int jpgdec_decode(jpgdec_t* d, const uint8_t* buf, int len){
    av_packet_unref(d->pkt);
    d->pkt->data = (uint8_t*)buf; d->pkt->size = len;
    if(avcodec_send_packet(d->ctx,d->pkt)<0) return -1;
    if(avcodec_receive_frame(d->ctx,d->frame)<0) return -1;
    if(!d->sws){
        d->sws = sws_getContext(d->frame->width,d->frame->height,d->frame->format,
                                IMG_N,IMG_N,AV_PIX_FMT_RGB24,SWS_BILINEAR,NULL,NULL,NULL);
        /* JPEG YUV 是 full range (0-255); 不显式声明 sws 按 limited 16-235 转 -> 整体偏色 */
        const int* coeffs = sws_getCoefficients(SWS_CS_DEFAULT);
        sws_setColorspaceDetails(d->sws, coeffs, 1 /* src full range */,
                                 coeffs, 1 /* dst full range */, 0, 1<<16, 1<<16);
    }
    uint8_t* dst[4]={d->rgb,NULL,NULL,NULL}; int dstStride[4]={IMG_N*3,0,0,0};
    sws_scale(d->sws,(const uint8_t* const*)d->frame->data,d->frame->linesize,0,d->frame->height,dst,dstStride);
    return 0;
}

/* rgb24 -> CHW f32 [0,255] (官方 JAX siglip 直接吃 0-255 原像素) */
static void rgb_to_chw(const uint8_t* rgb, float* out){
    const int N = IMG_N*IMG_N;
    for(int i=0;i<N;i++){
        out[i]        = (float)rgb[i*3+0] * (1.0f/127.5f) - 1.0f;
        out[N+i]      = (float)rgb[i*3+1] * (1.0f/127.5f) - 1.0f;
        out[2*N+i]    = (float)rgb[i*3+2] * (1.0f/127.5f) - 1.0f;
    }
}

/* ---------- npy 读(纯 f32, 自动跳 header) ---------- */
static float* npy_load_f32(const char* path, long* n_out){
    FILE* f = fopen(path,"rb");
    if(!f) return NULL;
    unsigned char hdr[8];
    if(fread(hdr,1,8,f)!=8){fclose(f);return NULL;}
    long off;
    if(hdr[6]==1){ unsigned short hl; if(fread(&hl,2,1,f)!=1){fclose(f);return NULL;} off = 8+2+hl; }
    else { unsigned int hl; if(fread(&hl,4,1,f)!=1){fclose(f);return NULL;} off = 8+4+hl; }
    fseek(f,0,SEEK_END); long total = ftell(f);
    long datasz = total - off;
    fseek(f,off,SEEK_SET);
    float* buf = malloc(datasz);
    if(fread(buf,1,datasz,f)!=(size_t)datasz){free(buf);fclose(f);return NULL;}
    fclose(f);
    *n_out = datasz/4;
    return buf;
}

/* npz 里取 float 数组(无压缩 npz=zip, 简单解析 local file header) */
static float* npz_load_f32(const char* path, const char* name, long* n_out){
    FILE* f = fopen(path,"rb");
    if(!f) return NULL;
    char want[256];
    snprintf(want,sizeof(want),"%s.npy",name);
    for(;;){
        unsigned char lh[30];
        if(fread(lh,1,30,f)!=30) break;
        if(memcmp(lh,"PK\x03\x04",4)!=0) break;
        unsigned short nlen = lh[26] | (lh[27]<<8);
        unsigned short elen = lh[28] | (lh[29]<<8);
        unsigned int csz = lh[18] | (lh[19]<<8) | (lh[20]<<16) | (lh[21]<<24);
        char nm[256] = {0};
        if(fread(nm,1,nlen,f)!=(size_t)nlen) break;
        fseek(f,elen,SEEK_CUR);
        if(strcmp(nm,want)==0){
            /* 读 npy header (zip data 起始处) */
            unsigned char m8[8];
            if(fread(m8,1,8,f)!=8) break;
            long hoff;
            if(m8[6]==1){ unsigned short hl; if(fread(&hl,2,1,f)!=1) break; hoff=hl; }
            else { unsigned int hl; if(fread(&hl,4,1,f)!=1) break; hoff=hl; }
            long datasz = csz - 10 - hoff;
            if(datasz<=0) break;
            fseek(f,hoff,SEEK_CUR);   /* 跳过 npy dict header (已消费 8B magic+ver, 2/4B hlen) */
            float* buf = malloc(datasz);
            if(fread(buf,1,datasz,f)!=(size_t)datasz){break;}
            fclose(f);
            *n_out = datasz/4;
            return buf;
        }
        fseek(f,csz,SEEK_CUR);
    }
    fclose(f);
    return NULL;
}

static void on_sigint(int s){ (void)s; if(g_srv>=0) close(g_srv); _exit(0); }

static double now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1e3 + ts.tv_nsec/1e6;
}

static pid_t g_child = -1;
static FILE* g_to_inf = NULL;
static FILE* g_from_inf = NULL;
static const char* g_siglip = "/data2/pi05s/demo_work/siglip_7a_F32.bmodel";
/* D/E 列切换: 默认 F32(D 列), 可用环境变量覆盖为 BF16(E 列) */
static const char* g_dkv0 = "/data2/pi05s/doffe/dkv0_9_bm1684x_F32.bmodel";
static const char* g_dkv1 = "/data2/pi05s/doffe/dkv9_18_bm1684x_F32.bmodel";
static const char* g_ddn0 = "/data2/pi05s/doffe/ddn_0_6_bm1684x_F32.bmodel";
static const char* g_ddn1 = "/data2/pi05s/doffe/ddn_6_12_bm1684x_F32.bmodel";
static const char* g_ddn2 = "/data2/pi05s/doffe/ddn_12_18_final_bm1684x_F32.bmodel";

static int spawn_child(int devid){
    if(g_to_inf){ fclose(g_to_inf); g_to_inf=NULL; }
    if(g_from_inf){ fclose(g_from_inf); g_from_inf=NULL; }
    if(g_child>0){ kill(g_child,SIGKILL); waitpid(g_child,NULL,0); }
    int in_pipe[2], out_pipe[2];
    if(pipe(in_pipe)||pipe(out_pipe)){perror("pipe");return 1;}
    pid_t pid = fork();
    if(pid==0){
        dup2(in_pipe[0],0); dup2(out_pipe[1],1);
        close(in_pipe[0]);close(in_pipe[1]);close(out_pipe[0]);close(out_pipe[1]);
        char dv[8]; snprintf(dv,sizeof(dv),"%d",devid);
        setenv("BMRUNTIME_NEURON_HEAP_MASK","7",1);
        /* H5: 子进程路径可由 PI05_SERVE_BIN 覆盖(532 验证服务与 968 生产服务并存) */
        const char* sbin = getenv("PI05_SERVE_BIN");
        if(!sbin) sbin = "/data2/pi05s/demo_work/opt2/serve_dkv";
        char* const av[] = {(char*)sbin, dv,
            (char*)g_siglip, (char*)g_dkv0, (char*)g_dkv1,
            (char*)g_ddn0, (char*)g_ddn1, (char*)g_ddn2, NULL};
        execv(av[0],av);
        _exit(127);
    }
    close(in_pipe[0]); close(out_pipe[1]);
    g_to_inf = fdopen(in_pipe[1],"wb");
    g_from_inf = fdopen(out_pipe[0],"rb");
    setvbuf(g_to_inf,NULL,_IONBF,0);
    setvbuf(g_from_inf,NULL,_IONBF,0);
    g_child = pid;
    fprintf(stderr,"infer child pid=%d (re)spawned\n",pid);
    return 0;
}

int main(int argc, char** argv){
    if(argc<4){fprintf(stderr,"usage: tcp_serve <devid> <workdir> <port> [siglip_bmodel]\n");return 2;}
    int devid = atoi(argv[1]);
    const char* WD = argv[2];
    int port = atoi(argv[3]);
    if(argc>=5) g_siglip = argv[4];
    {   /* D/E 列切换: 环境变量覆盖 dkv/ddn 路径 */
        const char* e;
        if((e=getenv("PI05_DKV0"))) g_dkv0=e;
        if((e=getenv("PI05_DKV1"))) g_dkv1=e;
        if((e=getenv("PI05_DDN0"))) g_ddn0=e;
        if((e=getenv("PI05_DDN1"))) g_ddn1=e;
        if((e=getenv("PI05_DDN2"))) g_ddn2=e;
    }
    fprintf(stderr,"siglip=%s\ndkv0=%s\ndkv1=%s\nddn0=%s\nddn1=%s\nddn2=%s\n",
            g_siglip,g_dkv0,g_dkv1,g_ddn0,g_ddn1,g_ddn2);
    signal(SIGINT,on_sigint); signal(SIGPIPE,SIG_IGN);

    if(spawn_child(devid)){fprintf(stderr,"spawn fail\n");return 1;}
    FILE* to_inf = g_to_inf;
    FILE* from_inf = g_from_inf;
    fprintf(stderr,"waiting LOADED...\n");
    sleep(20);  /* dkv+3ddn 加载约 20s */
    to_inf = g_to_inf; from_inf = g_from_inf;
    /* ---- JPEG 解码器 ---- */
    jpgdec_t dec;
    if(jpgdec_init(&dec)){fprintf(stderr,"jpgdec init fail\n");return 1;}

    /* ---- 常驻资产 ---- */
    long n_lang=0,n_dkva=0,n_un=0;
    float* lang200 = npy_load_f32("/data2/pi05s/demo_work/prompt200_t00.npy",&n_lang);
    long n_z3=0;
    float* z3h = npy_load_f32("/data2/pi05s/demo_work/z3_host.npy",&n_z3);  /* 黑图 feats 常量(host siglip 精确值, 替代设备 siglip 黑图 0.9919 偏差) */
    if(!lang200){fprintf(stderr,"assets load fail\n");return 1;}
    fprintf(stderr,"assets: lang=%ld\n",n_lang);
    /* dkv 资产(per-task npz): p_amask/f4d/p_cos/p_sin/s_cos/s_sin */
    long npam=0,nf4d=0,npcos=0,npsin=0,nscos=0,nssin=0;
    float* p_amask = npy_load_f32("/data2/pi05s/demo_work/dkva_npy/t00_p_amask.npy",&npam);
    float* f4d     = npy_load_f32("/data2/pi05s/demo_work/dkva_npy/t00_f4d.npy",&nf4d);
    float* p_cos   = npy_load_f32("/data2/pi05s/demo_work/dkva_npy/t00_p_cos.npy",&npcos);
    float* p_sin   = npy_load_f32("/data2/pi05s/demo_work/dkva_npy/t00_p_sin.npy",&npsin);
    float* s_cos   = npy_load_f32("/data2/pi05s/demo_work/dkva_npy/t00_s_cos.npy",&nscos);
    float* s_sin   = npy_load_f32("/data2/pi05s/demo_work/dkva_npy/t00_s_sin.npy",&nssin);
    if(!p_amask||!f4d||!p_cos||!p_sin||!s_cos||!s_sin){fprintf(stderr,"dkva load fail\n");return 1;}

    /* cond 权重 (action_in/time_mlp) + unnorm */
    float *W_in=NULL,*B_in=NULL,*W_ti=NULL,*B_ti=NULL,*W_to=NULL,*B_to=NULL,*mu=NULL,*sg=NULL;
    long q;
    W_in=npz_load_f32("/data2/pi05s/demo_work/cond_weights_libero.npz","action_in_proj_w",&q);
    B_in=npz_load_f32("/data2/pi05s/demo_work/cond_weights_libero.npz","action_in_proj_b",&q);
    W_ti=npz_load_f32("/data2/pi05s/demo_work/cond_weights_libero.npz","time_mlp_in_w",&q);
    B_ti=npz_load_f32("/data2/pi05s/demo_work/cond_weights_libero.npz","time_mlp_in_b",&q);
    W_to=npz_load_f32("/data2/pi05s/demo_work/cond_weights_libero.npz","time_mlp_out_w",&q);
    B_to=npz_load_f32("/data2/pi05s/demo_work/cond_weights_libero.npz","time_mlp_out_b",&q);
    mu =npz_load_f32("/data2/pi05s/demo_work/action_unnorm.npz","mean",&q);
    sg =npz_load_f32("/data2/pi05s/demo_work/action_unnorm.npz","std",&q);
    if(!W_in||!W_ti||!W_to||!mu){fprintf(stderr,"cond/unnorm load fail\n");return 1;}

    float* obs = malloc(3*(size_t)IMG_N*IMG_N*3*4);   /* 3 obs CHW f32 */
    float* feats = malloc(3*(size_t)256*2048*4);
    float* prefix = malloc((size_t)PL*2048*4);
    float* x = malloc((size_t)AH*32*4);
    float* vbuf = malloc((size_t)AH*32*4);
    float* xe = malloc((size_t)AH*1024*4);
    float* cd = malloc(1024*4);
    float* avec = malloc(1024*4);
    float frac[512], period[512], scal[512], emb[512*2];
    if(!obs||!feats||!prefix||!x||!vbuf||!xe||!cd||!avec){fprintf(stderr,"oom\n");return 1;}

    /* ---- TCP listen ---- */
    g_srv = socket(AF_INET,SOCK_STREAM,0);
    int opt=1; setsockopt(g_srv,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
    struct sockaddr_in addr; memset(&addr,0,sizeof(addr));
    addr.sin_family=AF_INET; addr.sin_addr.s_addr=htonl(INADDR_ANY); addr.sin_port=htons(port);
    if(bind(g_srv,(struct sockaddr*)&addr,sizeof(addr))<0){perror("bind");return 1;}
    if(listen(g_srv,4)<0){perror("listen");return 1;}
    fprintf(stderr,"TCP_SERVE_READY port=%d\n",port);

    uint8_t* jpgbuf = malloc(40u<<20);   /* 需容纳 250 模式 kv blob(35.7MB)+tidx=208 xp */
    int cur_task = -1;   /* 当前已载入的 per-task 资产对应的 task; -1=未载入。防每请求重载(见下) */
    for(;;){
        struct sockaddr_in cli; socklen_t cl=sizeof(cli);
        int cfd = accept(g_srv,(struct sockaddr*)&cli,&cl);
        if(cfd<0) continue;
        fprintf(stderr,"conn from %s\n",inet_ntoa(cli.sin_addr));
        /* 循环处理同一连接的多个请求 */
        for(;;){
            uint8_t hdr[18];
            { int got=0; while(got<18){ int k=read(cfd,hdr+got,18-got); if(k<=0){fprintf(stderr,"hdr read k=%d at %d\n",k,got); goto close_conn;} got+=k; } }
            uint8_t tidx = hdr[0], steps = hdr[1];
            int rgb_task = (tidx>=230 && tidx<240) ? (int)(tidx-230) : -1;  /* 200+task: RGB 无损 per-task 资产 */
            if(steps==0||steps>50) steps=10;
            int64_t seed; memcpy(&seed,hdr+2,8);
            uint32_t l0,l1; memcpy(&l0,hdr+10,4); memcpy(&l1,hdr+14,4);
            if(l0>(40u<<20)||l1>(24u<<20)) break;   /* 250 kv blob 35.7MB + 208 xp 需放宽 */
            { int got=0; while(got<AH*32){int k=read(cfd,((uint8_t*)x)+got,AH*32*4-got); if(k<=0){fprintf(stderr,"noise read k=%d\n",k);goto close_conn;} got+=k;} }
            int got=0; while(got<(int)l0){int k=read(cfd,jpgbuf+got,l0-got); if(k<=0)goto close_conn; got+=k;}
            uint8_t* jpg1 = jpgbuf+l0;
            got=0; while(got<(int)l1){int k=read(cfd,jpg1+got,l1-got); if(k<=0)goto close_conn; got+=k;}

            /* debug: tidx=206: 原始 RGB 无损输入(免 JPEG), 回传 feats */
            if(tidx==206){
                if(l0!=(uint32_t)IMG_N*IMG_N*3 || l1!=(uint32_t)IMG_N*IMG_N*3){
                    fprintf(stderr,"206 raw size %u %u\n",l0,l1); goto close_conn;
                }
                rgb_to_chw(jpgbuf, obs);
                rgb_to_chw(jpg1, obs + (size_t)IMG_N*IMG_N*3);
                float* z3 = obs + 2*(size_t)IMG_N*IMG_N*3;
                for(int zi=0; zi<IMG_N*IMG_N*3; zi++) z3[zi] = -1.0f;
                fputc('I',to_inf);
                fwrite(obs,4,3*IMG_N*IMG_N*3,to_inf);
                fflush(to_inf);
                for(int k=0;k<3;k++){
                    if(fread(feats + (size_t)k*256*2048,4,256*2048,from_inf)!=256*2048){
                        uint8_t e=0;write(cfd,&e,1);goto close_conn;
                    }
                }
                uint8_t ok=1; write(cfd,&ok,1);
                write(cfd,feats,3*256*2048*4);
                fprintf(stderr,"DBGRAW sent\n");
                continue;
            }
            /* debug: tidx=203 已废弃(旧资产协议) */
            if(tidx==203){ uint8_t e=0; write(cfd,&e,1); goto close_conn; }
            /* debug: tidx=202 -> 跳过 jpg 直接用 raw f32 obs? 不可行(协议只收jpg).
               改为 tidx=201: 用 PIL 等价解码不可行 — 直接回传 siglip 后的 feats 供 host 对比 */
            if(tidx==201){
                /* 正常走 I 命令, 但回传 feats 而非 traj */
                if(jpgdec_decode(&dec,jpgbuf,l0)){uint8_t e=0;write(cfd,&e,1);goto close_conn;}
                rgb_to_chw(dec.rgb, obs);
                if(jpgdec_decode(&dec,jpgbuf+l0,l1)){uint8_t e=0;write(cfd,&e,1);goto close_conn;}
                rgb_to_chw(dec.rgb, obs + (size_t)IMG_N*IMG_N*3);
                memset(obs + 2*(size_t)IMG_N*IMG_N*3, 0, (size_t)IMG_N*IMG_N*3*4);
                { float* z3 = obs + 2*(size_t)IMG_N*IMG_N*3; for(int zi=0; zi<IMG_N*IMG_N*3; zi++) z3[zi] = -1.0f; }
                fputc('I',to_inf);
                fwrite(obs,4,3*IMG_N*IMG_N*3,to_inf);
                fflush(to_inf);
                for(int k=0;k<3;k++){
                    if(fread(feats + (size_t)k*256*2048,4,256*2048,from_inf)!=256*2044*0+256*2048){uint8_t e=0;write(cfd,&e,1);goto close_conn;}
                }
                uint8_t ok=1; write(cfd,&ok,1);
                write(cfd,feats,3*256*2048*4);
                fprintf(stderr,"DBGFEAT sent\n");
                continue;
            }
            /* debug: tidx=200 -> 回传解码后 rgb (验证解码色彩正确性) */
            if(tidx==200){
                if(jpgdec_decode(&dec,jpgbuf,l0)){uint8_t e=0;write(cfd,&e,1);goto close_conn;}
                uint8_t ok=1; write(cfd,&ok,1);
                write(cfd,dec.rgb,IMG_N*IMG_N*3);
                fprintf(stderr,"DBGRGB sent\n");
                continue;
            }
            double t0=now_ms();
            /* 任务资产热切换(dkva_tXX.npz: p_amask/f4d/p_cos/p_sin/s_cos/s_sin)
             * 2026-09-14: 原实现**每个请求**都重载这 7 个文件(约 9.4MB malloc/free 抖动:
             * p_amask 3.75MB + 4 个 cos/sin 各 0.99MB + prompt200 1.64MB), 在只有 ~1GB
             * 系统 RAM 的设备上会把 serve_dkv 的 RSS 推到数百 MB 并触发 OOM
             * (实测每请求 +2.2MB, 2 case/34 次推理即 4MB→77MB)。
             * 这些资产只与 task 有关, 故按 task 缓存, task 不变就不重载。 */
            {
                int at = (rgb_task>=0) ? rgb_task : tidx;
                if(at>=0 && at<10 && at != cur_task){
                    char p[256];
                    float* v; char pp[300];
                    snprintf(pp,sizeof(pp),"/data2/pi05s/demo_work/dkva536_npy/t%02d_p_amask.npy",at);
                    v=npy_load_f32(pp,&q); if(v){free(p_amask);p_amask=v;}
                    snprintf(pp,sizeof(pp),"/data2/pi05s/demo_work/dkva536_npy/t%02d_f4d.npy",at);
                    v=npy_load_f32(pp,&q); if(v){free(f4d);f4d=v;}
                    snprintf(pp,sizeof(pp),"/data2/pi05s/demo_work/dkva536_npy/t%02d_p_cos.npy",at);
                    v=npy_load_f32(pp,&q); if(v){free(p_cos);p_cos=v;}
                    snprintf(pp,sizeof(pp),"/data2/pi05s/demo_work/dkva536_npy/t%02d_p_sin.npy",at);
                    v=npy_load_f32(pp,&q); if(v){free(p_sin);p_sin=v;}
                    snprintf(pp,sizeof(pp),"/data2/pi05s/demo_work/dkva536_npy/t%02d_s_cos.npy",at);
                    v=npy_load_f32(pp,&q); if(v){free(s_cos);s_cos=v;}
                    snprintf(pp,sizeof(pp),"/data2/pi05s/demo_work/dkva536_npy/t%02d_s_sin.npy",at);
                    v=npy_load_f32(pp,&q); if(v){free(s_sin);s_sin=v;}
                    snprintf(p,sizeof(p),"/data2/pi05s/demo_work/dkva536_npy/promptL_t%02d.npy",at);
                    float* nl = npy_load_f32(p,&n_lang); if(nl){free(lang200);lang200=nl;}
                    cur_task = at;
                    fprintf(stderr,"task assets loaded: t%02d\n",at);
                }
            }

            /* I: 3 obs (jpg0, jpg1 解码 + zeros) -> siglip -> feats */
            if(tidx==250){ /* host-KV 模式: 请求体携带 kv_server 产物(36KV+f4d+s_cos+s_sin), 转发 serve_dkv K+S */
                size_t want_kv = (size_t)36*968*256*4 + (size_t)AH*978*4 + (size_t)AH*256*4*2;
                if(l0!=(uint32_t)want_kv){fprintf(stderr,"250 size %u vs %zu\n",l0,want_kv);goto close_conn;}
                /* 转发 K: 资产(f4d,s_cos,s_sin 在 blob 中的偏移 36*968*256*4 之后) + 36 KV */
                /* blob 布局: [36 KV(各 968*256 f32)][f4d AH*978][s_cos AH*256][s_sin AH*256] */
                float* blob = (float*)jpgbuf;
                float* kvp = blob;
                float* bf4d = blob + (size_t)36*968*256;
                float* bsc   = bf4d + (size_t)AH*978;
                float* bss   = bsc + (size_t)AH*256;
                memcpy(f4d, bf4d, (size_t)AH*978*4);
                memcpy(s_cos, bsc, (size_t)AH*256*4);
                memcpy(s_sin, bss, (size_t)AH*256*4);
                /* p_amask 仍从 at 任务资产读(serve_dkv K 需要它? K 协议含 p_amask) */
                int at250 = (int)(*(float*)jpg1);
                if(at250>=0 && at250<10){
                    char p250[256];
                    snprintf(p250,sizeof(p250),"/data2/pi05s/demo_work/dkva536_npy/t%02d_p_amask.npy",at250);
                    float* v=npy_load_f32(p250,&q); if(v){free(p_amask);p_amask=v;}
                }
                /* K: p_amask + f4d + s_cos + s_sin + 36 KV */
                fputc('K',to_inf);
                fwrite(p_amask,4,PL*PL,to_inf);
                fwrite(f4d,4,AH*(PL+AH),to_inf);
                fwrite(s_cos,4,AH*256,to_inf);
                fwrite(s_sin,4,AH*256,to_inf);
                fwrite(kvp,4,36*(size_t)968*256,to_inf);
                fflush(to_inf);
                char okk[2];
                if(fread(okk,1,2,from_inf)!=2||okk[0]!='O'){fprintf(stderr,"250 K fail\n");goto err;}
                goto s250_run;
            }
            if(tidx==240){ /* host-prefix 模式: 请求体直接携带 prefix(968x2048 f32), 跳过 siglip, 直接 P'+Euler */
                if(l0!=(uint32_t)PL*2048*4){fprintf(stderr,"240 size %u\n",l0);goto close_conn;}
                memcpy(prefix, jpgbuf, (size_t)PL*2048*4);
                /* 任务资产热切换(at 由 seed 字段传递? 用 noise 区? 简化: 240 模式 at 固定读 jpg1 首字节) */
                int at240 = (int)(*(float*)jpg1);
                if(at240>=0 && at240<10){
                    char p240[256];
                    snprintf(p240,sizeof(p240),"/data2/pi05s/demo_work/dkva536_npy/t%02d_p_amask.npy",at240);
                    float* v=npy_load_f32(p240,&q); if(v){free(p_amask);p_amask=v;}
                    snprintf(p240,sizeof(p240),"/data2/pi05s/demo_work/dkva536_npy/t%02d_f4d.npy",at240);
                    v=npy_load_f32(p240,&q); if(v){free(f4d);f4d=v;}
                    snprintf(p240,sizeof(p240),"/data2/pi05s/demo_work/dkva536_npy/t%02d_p_cos.npy",at240);
                    v=npy_load_f32(p240,&q); if(v){free(p_cos);p_cos=v;}
                    snprintf(p240,sizeof(p240),"/data2/pi05s/demo_work/dkva536_npy/t%02d_p_sin.npy",at240);
                    v=npy_load_f32(p240,&q); if(v){free(p_sin);p_sin=v;}
                    snprintf(p240,sizeof(p240),"/data2/pi05s/demo_work/dkva536_npy/t%02d_s_cos.npy",at240);
                    v=npy_load_f32(p240,&q); if(v){free(s_cos);s_cos=v;}
                    snprintf(p240,sizeof(p240),"/data2/pi05s/demo_work/dkva536_npy/t%02d_s_sin.npy",at240);
                    v=npy_load_f32(p240,&q); if(v){free(s_sin);s_sin=v;}
                }
                goto p240_run;
            }
            if(tidx==209){ /* debug: raw 2图 -> siglip -> prefix(z3h+lang) 回传 */
                if(l0!=(uint32_t)IMG_N*IMG_N*3 || l1!=(uint32_t)IMG_N*IMG_N*3){fprintf(stderr,"209 raw size %u %u\n",l0,l1);goto close_conn;}
                /* H5 修复: 原实现按 968 布局写 prefix(768/512/768 偏移), 在 PL=536 下
                 * 会越过 prefix 缓冲区(532*2048 floats)造成堆溢出并令子进程崩溃。
                 * 现与主路径同构: 只跑 2 路 siglip, 语言取 n_lang, 余下补位槽填 0
                 * (运行时 p_amask 会把补位槽整段屏蔽, 取值不影响输出)。 */
                rgb_to_chw(jpgbuf, obs);
                rgb_to_chw(jpg1, obs + (size_t)IMG_N*IMG_N*3);
                fputc('J',to_inf); fwrite(obs,4,2*(size_t)IMG_N*IMG_N*3,to_inf); fflush(to_inf);
                for(int k=0;k<2;k++){ if(fread(feats + (size_t)k*256*2048,4,256*2048,from_inf)!=256*2048){uint8_t e=0;write(cfd,&e,1);goto close_conn;} }
                memcpy(prefix, feats, 512*2048*4);
                {
                    long nl = n_lang / 2048;
                    if(nl > (PL - 512)) nl = PL - 512;
                    memcpy(prefix + (size_t)512*2048, lang200, (size_t)nl*2048*4);
                    if(nl < (PL - 512))
                        memset(prefix + (size_t)(512+nl)*2048, 0, (size_t)(PL-512-nl)*2048*4);
                }
                uint8_t ok=1; write(cfd,&ok,1); write(cfd,prefix,PL*2048*4);
                fprintf(stderr,"DBG209 prefix sent\n"); continue;
            }
            if (rgb_task>=0 || tidx==207) {
                if(l0!=(uint32_t)IMG_N*IMG_N*3 || l1!=(uint32_t)IMG_N*IMG_N*3){fprintf(stderr,"207 raw size %u %u\n",l0,l1);goto close_conn;}
                rgb_to_chw(jpgbuf, obs);
                rgb_to_chw(jpg1, obs + (size_t)IMG_N*IMG_N*3);
            } else {
                if(jpgdec_decode(&dec,jpgbuf,l0)){fprintf(stderr,"dec0 fail\n");goto err;}
                rgb_to_chw(dec.rgb, obs);
                if(jpgdec_decode(&dec,jpg1,l1)){fprintf(stderr,"dec1 fail\n");goto err;}
                rgb_to_chw(dec.rgb, obs + (size_t)IMG_N*IMG_N*3);
            }
            /* 2026-09-14 性能优化: 第 3 路是补零黑图, 其 siglip 结果紧接着就被下面的
             * z3h 常量整段覆盖 —— 设备为此白跑 37ms/次推理. 有 z3h 时改发 'J'(2 图),
             * 只跑前两路; prefix 的 [512:768] 段直接由 z3h 填充, 数值路径不变.
             * z3h 缺失时回退到原来的 3 图 'I' 路径, 行为与优化前逐位一致. */
            /* H5: prefix = 2 路真实图像 feats(512 tok) + 语言(token 数随 task 变, 由 prompt20 提供)
             * 第 3 路黑图的 256 个 token 与语言 padding 已整段删除(证明见 docs/h5_dead_token_20260914.md) */
            fputc('J',to_inf);
            fwrite(obs,4,2*(size_t)IMG_N*IMG_N*3,to_inf);
            fflush(to_inf);
            for(int k=0;k<2;k++){
                if(fread(feats + (size_t)k*256*2048,4,256*2048,from_inf)!=256*2048){fprintf(stderr,"feat read fail\n");goto err;}
            }

            /* prefix = feats(512 tok) + lang(n_lang tok, 其余为已屏蔽的 padding) */
            memcpy(prefix, feats, 512*2048*4);
            {
                long nl = n_lang / 2048;
                if(nl > (PL - 512)) nl = PL - 512;
                memcpy(prefix + (size_t)512*2048, lang200, (size_t)nl*2048*4);
                /* 剩余槽位填 0 即可: p_amask 已把它们整段屏蔽, 取值不影响输出 */
                if(nl < (PL - 512))
                    memset(prefix + (size_t)(512+nl)*2048, 0, (size_t)(PL-512-nl)*2048*4);
            }
            double t1=now_ms();

            /* P': prefix + 6 资产 + time(初始) -> 设备跑 dkv(36 KV 常驻) */
        p240_run: ;
            (void)0;
        s250_run: ;
            fputc('P',to_inf);
            fwrite(prefix,4,PL*2048,to_inf);
            fwrite(p_amask,4,PL*PL,to_inf);
            fwrite(p_cos,4,PL*256,to_inf);
            fwrite(p_sin,4,PL*256,to_inf);
            fwrite(f4d,4,AH*(size_t)(PL+AH),to_inf);
            fwrite(s_cos,4,AH*256,to_inf);
            fwrite(s_sin,4,AH*256,to_inf);
            { float tinit=1.0f; fwrite(&tinit,4,1,to_inf); }
            fflush(to_inf);
            char ok2[2];
            if(fread(ok2,1,2,from_inf)!=2||ok2[0]!='O'){fprintf(stderr,"P fail\n");goto err;}
            double t2=now_ms();

            (void)seed;   /* noise 由 client 随请求下发(1280B), host 端掌控 seed 语义 */
            float dt = -1.0f/steps;
            for(int st=0; st<steps; st++){
                float t = 1.0f + st*dt;
                /* S': x_t + time(每步) -> 设备 ddn 链 -> v_t */
                fputc('S',to_inf);
                fwrite(x,4,AH*32,to_inf);
                fwrite(&t,4,1,to_inf);
                fflush(to_inf);
                if(fread(vbuf,4,AH*32,from_inf)!=AH*32){fprintf(stderr,"S read fail\n");goto err;}
                if(st==0) fprintf(stderr,"DBG v[0:4]=%.3f %.3f %.3f %.3f\n",vbuf[0],vbuf[1],vbuf[2],vbuf[3]);
                for(int i=0;i<AH*32;i++) x[i] += dt*vbuf[i];
            }
            double t3=now_ms();
            fprintf(stderr,"REQ tidx=%d steps=%d dec+sig=%.1fms P=%.1fms S=%.1fms total=%.1fms\n",
                    tidx,steps,t1-t0,t2-t1,t3-t2,t3-t0);
            /* unnorm 前 7 维 */
            for(int j=0;j<AH;j++)
                for(int d=0;d<7;d++)
                    x[j*32+d] = x[j*32+d]*sg[d] + mu[d];
            /* 2026-09-15: ok(1B)+traj(1280B) 合并成一次 write, 并删掉原来的
             * usleep(200000)。原实现分两次 write 后睡 200ms, 是为绕开客户端
             * _recv_exact 丢余量的 bug(ok 与 traj 同段到达时 traj 被丢弃);
             * 该 bug 已在客户端侧根治(内部缓冲), 合并写又让两个字段必然同段到达。
             * 闭环里这 200ms 被客户端仿真掩盖, 背靠背/多任务场景是实打实的 1.44x。 */
            {   uint8_t resp[1 + TRAJ_B];
                resp[0] = 1;
                memcpy(resp + 1, x, TRAJ_B);
                size_t off = 0;
                while (off < sizeof(resp)) {
                    ssize_t w = write(cfd, resp + off, sizeof(resp) - off);
                    if (w <= 0) { fprintf(stderr, "RESP write fail at %zu\n", off); break; }
                    off += (size_t)w;
                }
            }
            continue;
        err:
            {uint8_t ok=0; write(cfd,&ok,1);}
        }
close_conn:
        close(cfd);
        {   /* 子进程探活: 死了重启 */
            int st; pid_t r = waitpid(-1, &st, WNOHANG);
            if(r == g_child || (r < 0 && errno == ECHILD)) {
                fprintf(stderr,"child %d exited, respawning\n", g_child);
                if(spawn_child(devid)==0){ sleep(20); }
                to_inf = g_to_inf; from_inf = g_from_inf;
            }
        }
    }
    return 0;
}
