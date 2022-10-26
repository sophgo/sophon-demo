//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
/*
This code is refer from: 
https://github.com/AlexeyAB/darknet/blob/master/src/parser.c
https://github.com/AlexeyAB/darknet/blob/master/src/list.c
https://github.com/AlexeyAB/darknet/blob/master/src/option_list.c
https://github.com/AlexeyAB/darknet/blob/master/src/utils.c
*/
#ifndef _YOLACT_PARSER_HPP_
#define _YOLACT_PARSER_HPP_

#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
using namespace std;

#ifdef __cplusplus
extern "C" {
#endif

#define PARSER_LOC __FILE__, __func__, __LINE__

const char * size_to_IEC_string(const size_t size)
{
    const float bytes = (double)size;
    const float KiB = 1024;
    const float MiB = 1024 * KiB;
    const float GiB = 1024 * MiB;

    static char buffer[25];
    if (size < KiB)         sprintf(buffer, "%ld bytes", size);
    else if (size < MiB)    sprintf(buffer, "%1.1f KiB", bytes / KiB);
    else if (size < GiB)    sprintf(buffer, "%1.1f MiB", bytes / MiB);
    else                    sprintf(buffer, "%1.1f GiB", bytes / GiB);

    return buffer;
}

void xerror(const char * const msg, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Darknet error location: %s, %s, line #%d\n", filename, funcname, line);
    perror(msg);
    exit(EXIT_FAILURE);
}

void malloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Failed to malloc %s\n", size_to_IEC_string(size));
    xerror("Malloc error - possibly out of CPU RAM", filename, funcname, line);
}

void calloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Failed to calloc %s\n", size_to_IEC_string(size));
    xerror("Calloc error - possibly out of CPU RAM", filename, funcname, line);
}

void realloc_error(const size_t size, const char * const filename, const char * const funcname, const int line)
{
    fprintf(stderr, "Failed to realloc %s\n", size_to_IEC_string(size));
    xerror("Realloc error - possibly out of CPU RAM", filename, funcname, line);
}

void file_error(const char * const s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(EXIT_FAILURE);
}


void *xmalloc_location(const size_t size, const char * const filename, const char * const funcname, const int line) {
    void *ptr=malloc(size);
    if(!ptr) {
        malloc_error(size, filename, funcname, line);
    }
    return ptr;
}

void *xcalloc_location(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line) {
    void *ptr=calloc(nmemb, size);
    if(!ptr) {
        calloc_error(nmemb * size, filename, funcname, line);
    }
    return ptr;
}

void *xrealloc_location(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line) {
    ptr=realloc(ptr,size);
    if(!ptr) {
        realloc_error(size, filename, funcname, line);
    }
    return ptr;
}



#define xmalloc(s)      xmalloc_location(s, PARSER_LOC)
#define xcalloc(m, s)   xcalloc_location(m, s, PARSER_LOC)
#define xrealloc(p, s)  xrealloc_location(p, s, PARSER_LOC)


void error(const char *s){
    fprintf(stderr, "Error: %s\n", s);
    exit(-1);
}

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list_t{
    int size;
    node *front;
    node *back;
} list_t;


list_t *make_list()
{
    list_t* l = (list_t*)xmalloc(sizeof(list_t));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

/*
void transfer_node(list_t *s, list_t *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list_t *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;

    return val;
}

void list_insert(list_t *l, void *val)
{
    node* newnode = (node*)xmalloc(sizeof(node));
    newnode->val = val;
    newnode->next = 0;

    if(!l->back){
        l->front = newnode;
        newnode->prev = 0;
    }else{
        l->back->next = newnode;
        newnode->prev = l->back;
    }
    l->back = newnode;
    ++l->size;
}

void free_node(node *n)
{
    node *next;
    while(n) {
        next = n->next;
        free(n);
        n = next;
    }
}

void free_list_val(list_t *l)
{
    node *n = l->front;
    node *next;
    while (n) {
        next = n->next;
        free(n->val);
        n = next;
    }
}

void free_list(list_t *l)
{
    free_node(l->front);
    free(l);
}

void free_list_contents(list_t *l)
{
    node *n = l->front;
    while(n){
        free(n->val);
        n = n->next;
    }
}

void free_list_contents_kvp(list_t *l)
{
    node *n = l->front;
    while (n) {
        kvp* p = (kvp*)n->val;
        free(p->key);
        free(n->val);
        n = n->next;
    }
}

void **list_to_array(list_t *l)
{
    void** a = (void**)xcalloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}


typedef struct{
    char *type;
    list_t *options;
}section;


void option_insert(list_t *l, char *key, char *val)
{
    kvp* p = (kvp*)xmalloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}


int read_option(char *s, list_t *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char* line = (char*)xmalloc(size * sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);
    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = (char*)xrealloc(line, size * sizeof(char));
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        if (fgets(&line[curr], readsize, fp) != NULL){
            curr = strlen(line);
        }
    }
    if(curr >= 2)
        if(line[curr-2] == 0x0d) line[curr-2] = 0x00;

    if(curr >= 1)
        if(line[curr-1] == 0x0a) line[curr-1] = 0x00;

    return line;
}


void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n'||c =='\r'||c==0x0d||c==0x0a) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}


list_t *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list_t *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = (section*)xmalloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

int is_network(section *s)
{
    return (strcmp(s->type, "[network]")==0
            || strcmp(s->type, "[params]")==0);
}

char *option_find(list_t *l, const char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}

char *option_find_str(list_t *l, const char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

char *option_find_str_quiet(list_t *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if (v) return v;
    return def;
}

int option_find_int(list_t *l, const char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

int option_find_int_quiet(list_t *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list_t *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list_t *l, const char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}


int *parse_int_list(char *a, int *num) 
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        //mask = (int *)calloc(n, sizeof(int));
        mask = new int[n];
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

float *parse_float_list(char *a, int *num) 
{
    float *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        //mask = (float *)calloc(n, sizeof(float));
        mask = new float[n];
        for (i = 0; i < n; ++i) {
            float val = atof(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

# if 0
// https://github.com/AlexeyAB/darknet/blob/master/src/parser.c#L1360
void parse_cfg_test(char *filename)
{
    // /*************************/
    // [yolact]
    // normalize = 1
    // subtract_means = 0
    // to_float = 0

    // num_classes = 80
    // width = 550
    // height = 550
    // num_scales = 5
    // num_aspect_ratios = 3
    // conv_ws = 69, 35, 18, 9, 5
    // conv_hs = 69, 35, 18, 9, 5
    // scales = 24, 48, 96, 192, 384
    // aspect_ratios = 1, 0.5, 2
    // variances = 0.1, 0.2

    // [postprocess]
    // thresh = 0.5
    // /*************************/

    
    list_t *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");

    section *s = (section *)n->val;
    list_t *options = s->options;
    if(!is_network(s)) error("First section must be [yolact] or [network]");

    int normalize = option_find_int(options, "normalize", 1);
    int subtract_means = option_find_int(options, "subtract_means", 0);
    int to_float = option_find_int(options, "to_float", 0);

    int width = option_find_int(options, "width", 550);
    int height = option_find_int(options, "height", 550);

    int num_scales = option_find_int(options, "num_scales", 5);
    int num_aspect_ratios = option_find_int(options, "num_aspect_ratios", 3);

    char *a = option_find_str(options, "conv_ws", 0);
    int *conv_ws = parse_int_list(a, &num_scales);

    char *a1 = option_find_str(options, "aspect_ratios", 0);
    float *aspect_ratios = parse_float_list(a1, &num_aspect_ratios);


    free(conv_ws);
    free(aspect_ratios);

    free_section(s);

    n = n->next;

    s = (section *)n->val;
    options = s->options;
    float thresh = option_find_float(options, "thresh", 0.6);
    std::cout << "sec2: thresh ==> " << thresh << std::endl;

    free_section(s);

    free_list(sections);
    

}
#endif

#ifdef __cplusplus
}
#endif


#endif