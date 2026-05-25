//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "vits_preprocess.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

VitsPreprocessor::VitsPreprocessor(const string& bert_model,
                                     const string& pinyin_map_file,
                                     const string& vocab_file,
                                     int dev_id) {
    // build pinyin_dict and symbol table
    build_pinyin_dict();
    build_symbol_table();

    // load data files
    load_pinyin_map(pinyin_map_file);
    load_vocab(vocab_file);

    // get handle
    auto ret = bm_dev_request(&handle, dev_id);
    assert(BM_SUCCESS == ret);

    // judge pcie or soc
    ret = bm_get_misc_info(handle, &misc_info);
    assert(BM_SUCCESS == ret);

    // create bmrt
    bmrt = bmrt_create(handle);
    if (!bmrt_load_bmodel(bmrt, bert_model.c_str())) {
        cout << "load bmodel(" << bert_model << ") failed" << endl;
        exit(1);
    }

    // get network names
    const char **names;
    int num = bmrt_get_network_number(bmrt);
    bmrt_get_network_names(bmrt, &names);
    for (int i = 0; i < num; ++i) {
        network_names.push_back(names[i]);
    }
    free(names);

    // get netinfo
    netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());

    // get input/output shapes
    m_max_length = netinfo->stages[0].input_shapes[0].dims[1];
    m_embed_dim = netinfo->stages[0].output_shapes[0].dims[2];

    cout << "BERT max_length: " << m_max_length << ", embed_dim: " << m_embed_dim << endl;

    m_ts = &tmp_ts;
}

VitsPreprocessor::~VitsPreprocessor() {
    if (bmrt != NULL) {
        bmrt_destroy(bmrt);
        bmrt = NULL;
    }
    bm_dev_free(handle);
}

float* VitsPreprocessor::get_cpu_data(bm_tensor_t* tensor) {
    int count = bmrt_shape_count(&tensor->shape);
    float* pFP32 = new float[count];
    assert(pFP32 != NULL);

    if (misc_info.pcie_soc_mode == 1) {
        unsigned long long addr;
        bm_mem_mmap_device_mem(handle, &tensor->device_mem, &addr);
        bm_mem_invalidate_device_mem(handle, &tensor->device_mem);
        memcpy(pFP32, (float*)addr, count * sizeof(float));
        bm_mem_unmap_device_mem(handle, (float*)addr, bm_mem_get_device_size(tensor->device_mem));
    } else {
        bm_memcpy_d2s_partial(handle, pFP32, tensor->device_mem, count * sizeof(float));
    }
    return pFP32;
}

bool VitsPreprocessor::is_chinese(uint32_t cp) {
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

// Decode UTF-8 character, return code point. Returns number of bytes consumed.
static int utf8_decode(const string& s, size_t pos, uint32_t& cp) {
    unsigned char c = s[pos];
    if (c < 0x80) {
        cp = c;
        return 1;
    } else if (c < 0xE0) {
        cp = ((c & 0x1F) << 6) | (s[pos + 1] & 0x3F);
        return 2;
    } else if (c < 0xF0) {
        cp = ((c & 0x0F) << 12) | ((s[pos + 1] & 0x3F) << 6) | (s[pos + 2] & 0x3F);
        return 3;
    } else {
        cp = ((c & 0x07) << 18) | ((s[pos + 1] & 0x3F) << 12) |
             ((s[pos + 2] & 0x3F) << 6) | (s[pos + 3] & 0x3F);
        return 4;
    }
}

// Encode a code point to UTF-8 string
static string utf8_encode(uint32_t cp) {
    if (cp < 0x80) {
        return string(1, (char)cp);
    } else if (cp < 0x800) {
        char buf[3] = {(char)(0xC0 | (cp >> 6)), (char)(0x80 | (cp & 0x3F)), 0};
        return string(buf);
    } else {
        char buf[4] = {(char)(0xE0 | (cp >> 12)),
                       (char)(0x80 | ((cp >> 6) & 0x3F)),
                       (char)(0x80 | (cp & 0x3F)), 0};
        return string(buf);
    }
}

string VitsPreprocessor::clean_chinese(const string& text) {
    string text_clean;
    for (size_t i = 0; i < text.size();) {
        uint32_t cp;
        int len = utf8_decode(text, i, cp);
        if (is_chinese(cp)) {
            text_clean += text.substr(i, len);
        } else {
            if (text_clean.size() > 1) {
                // Check if last char was Chinese
                size_t last_start = text_clean.size();
                while (last_start > 0 && (unsigned char)text_clean[last_start - 1] >= 0x80 &&
                       (unsigned char)text_clean[last_start - 1] < 0xC0)
                    last_start--;
                // Just check: if we have at least one Chinese char before
                text_clean += ',';
            }
        }
        i += len;
    }
    // Trim trailing commas
    while (!text_clean.empty() && text_clean.back() == ',')
        text_clean.pop_back();
    return text_clean;
}

string VitsPreprocessor::char_to_pinyin(const string& ch) {
    auto it = m_pinyin_map.find(ch);
    if (it != m_pinyin_map.end()) {
        return it->second;
    }
    return "";
}

vector<string> VitsPreprocessor::get_phoneme4pinyin(const vector<string>& pinyins,
                                                       vector<int>& count_phone) {
    vector<string> result;
    count_phone.clear();
    for (const auto& pinyin : pinyins) {
        if (pinyin.empty()) continue;
        // last char is tone, rest is pinyin base
        string tone(1, pinyin.back());
        string base = pinyin.substr(0, pinyin.size() - 1);
        auto it = m_pinyin_dict.find(base);
        if (it != m_pinyin_dict.end()) {
            result.push_back(it->second.first);         // initial
            result.push_back(it->second.second + tone); // final + tone
            count_phone.push_back(2);
        }
    }
    return result;
}

void VitsPreprocessor::load_pinyin_map(const string& filepath) {
    ifstream f(filepath);
    if (!f.is_open()) {
        cerr << "Cannot open pinyin map: " << filepath << endl;
        exit(1);
    }
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        size_t space = line.find(' ');
        if (space != string::npos) {
            string hanzi = line.substr(0, space);
            string pinyin = line.substr(space + 1);
            // Trim trailing newline
            while (!pinyin.empty() && (pinyin.back() == '\r' || pinyin.back() == '\n'))
                pinyin.pop_back();
            m_pinyin_map[hanzi] = pinyin;
        }
    }
    cout << "Loaded " << m_pinyin_map.size() << " pinyin mappings" << endl;
}

void VitsPreprocessor::load_vocab(const string& filepath) {
    ifstream f(filepath);
    if (!f.is_open()) {
        cerr << "Cannot open vocab: " << filepath << endl;
        exit(1);
    }
    string line;
    int id = 0;
    while (getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        m_vocab[line] = id++;
    }
    cout << "Loaded " << m_vocab.size() << " BERT vocab entries" << endl;
}

void VitsPreprocessor::build_pinyin_dict() {
    // Auto-generated from Python pinyin_dict
    m_pinyin_dict["a"] = {"^", "a"};
    m_pinyin_dict["ai"] = {"^", "ai"};
    m_pinyin_dict["an"] = {"^", "an"};
    m_pinyin_dict["ang"] = {"^", "ang"};
    m_pinyin_dict["ao"] = {"^", "ao"};
    m_pinyin_dict["ba"] = {"b", "a"};
    m_pinyin_dict["bai"] = {"b", "ai"};
    m_pinyin_dict["ban"] = {"b", "an"};
    m_pinyin_dict["bang"] = {"b", "ang"};
    m_pinyin_dict["bao"] = {"b", "ao"};
    m_pinyin_dict["be"] = {"b", "e"};
    m_pinyin_dict["bei"] = {"b", "ei"};
    m_pinyin_dict["ben"] = {"b", "en"};
    m_pinyin_dict["beng"] = {"b", "eng"};
    m_pinyin_dict["bi"] = {"b", "i"};
    m_pinyin_dict["bian"] = {"b", "ian"};
    m_pinyin_dict["biao"] = {"b", "iao"};
    m_pinyin_dict["bie"] = {"b", "ie"};
    m_pinyin_dict["bin"] = {"b", "in"};
    m_pinyin_dict["bing"] = {"b", "ing"};
    m_pinyin_dict["bo"] = {"b", "o"};
    m_pinyin_dict["bu"] = {"b", "u"};
    m_pinyin_dict["ca"] = {"c", "a"};
    m_pinyin_dict["cai"] = {"c", "ai"};
    m_pinyin_dict["can"] = {"c", "an"};
    m_pinyin_dict["cang"] = {"c", "ang"};
    m_pinyin_dict["cao"] = {"c", "ao"};
    m_pinyin_dict["ce"] = {"c", "e"};
    m_pinyin_dict["cen"] = {"c", "en"};
    m_pinyin_dict["ceng"] = {"c", "eng"};
    m_pinyin_dict["cha"] = {"ch", "a"};
    m_pinyin_dict["chai"] = {"ch", "ai"};
    m_pinyin_dict["chan"] = {"ch", "an"};
    m_pinyin_dict["chang"] = {"ch", "ang"};
    m_pinyin_dict["chao"] = {"ch", "ao"};
    m_pinyin_dict["che"] = {"ch", "e"};
    m_pinyin_dict["chen"] = {"ch", "en"};
    m_pinyin_dict["cheng"] = {"ch", "eng"};
    m_pinyin_dict["chi"] = {"ch", "iii"};
    m_pinyin_dict["chong"] = {"ch", "ong"};
    m_pinyin_dict["chou"] = {"ch", "ou"};
    m_pinyin_dict["chu"] = {"ch", "u"};
    m_pinyin_dict["chua"] = {"ch", "ua"};
    m_pinyin_dict["chuai"] = {"ch", "uai"};
    m_pinyin_dict["chuan"] = {"ch", "uan"};
    m_pinyin_dict["chuang"] = {"ch", "uang"};
    m_pinyin_dict["chui"] = {"ch", "uei"};
    m_pinyin_dict["chun"] = {"ch", "uen"};
    m_pinyin_dict["chuo"] = {"ch", "uo"};
    m_pinyin_dict["ci"] = {"c", "ii"};
    m_pinyin_dict["cong"] = {"c", "ong"};
    m_pinyin_dict["cou"] = {"c", "ou"};
    m_pinyin_dict["cu"] = {"c", "u"};
    m_pinyin_dict["cuan"] = {"c", "uan"};
    m_pinyin_dict["cui"] = {"c", "uei"};
    m_pinyin_dict["cun"] = {"c", "uen"};
    m_pinyin_dict["cuo"] = {"c", "uo"};
    m_pinyin_dict["da"] = {"d", "a"};
    m_pinyin_dict["dai"] = {"d", "ai"};
    m_pinyin_dict["dan"] = {"d", "an"};
    m_pinyin_dict["dang"] = {"d", "ang"};
    m_pinyin_dict["dao"] = {"d", "ao"};
    m_pinyin_dict["de"] = {"d", "e"};
    m_pinyin_dict["dei"] = {"d", "ei"};
    m_pinyin_dict["den"] = {"d", "en"};
    m_pinyin_dict["deng"] = {"d", "eng"};
    m_pinyin_dict["di"] = {"d", "i"};
    m_pinyin_dict["dia"] = {"d", "ia"};
    m_pinyin_dict["dian"] = {"d", "ian"};
    m_pinyin_dict["diao"] = {"d", "iao"};
    m_pinyin_dict["die"] = {"d", "ie"};
    m_pinyin_dict["ding"] = {"d", "ing"};
    m_pinyin_dict["diu"] = {"d", "iou"};
    m_pinyin_dict["dong"] = {"d", "ong"};
    m_pinyin_dict["dou"] = {"d", "ou"};
    m_pinyin_dict["du"] = {"d", "u"};
    m_pinyin_dict["duan"] = {"d", "uan"};
    m_pinyin_dict["dui"] = {"d", "uei"};
    m_pinyin_dict["dun"] = {"d", "uen"};
    m_pinyin_dict["duo"] = {"d", "uo"};
    m_pinyin_dict["e"] = {"^", "e"};
    m_pinyin_dict["ei"] = {"^", "ei"};
    m_pinyin_dict["en"] = {"^", "en"};
    m_pinyin_dict["ng"] = {"^", "en"};
    m_pinyin_dict["eng"] = {"^", "eng"};
    m_pinyin_dict["er"] = {"^", "er"};
    m_pinyin_dict["fa"] = {"f", "a"};
    m_pinyin_dict["fan"] = {"f", "an"};
    m_pinyin_dict["fang"] = {"f", "ang"};
    m_pinyin_dict["fei"] = {"f", "ei"};
    m_pinyin_dict["fen"] = {"f", "en"};
    m_pinyin_dict["feng"] = {"f", "eng"};
    m_pinyin_dict["fo"] = {"f", "o"};
    m_pinyin_dict["fou"] = {"f", "ou"};
    m_pinyin_dict["fu"] = {"f", "u"};
    m_pinyin_dict["ga"] = {"g", "a"};
    m_pinyin_dict["gai"] = {"g", "ai"};
    m_pinyin_dict["gan"] = {"g", "an"};
    m_pinyin_dict["gang"] = {"g", "ang"};
    m_pinyin_dict["gao"] = {"g", "ao"};
    m_pinyin_dict["ge"] = {"g", "e"};
    m_pinyin_dict["gei"] = {"g", "ei"};
    m_pinyin_dict["gen"] = {"g", "en"};
    m_pinyin_dict["geng"] = {"g", "eng"};
    m_pinyin_dict["gong"] = {"g", "ong"};
    m_pinyin_dict["gou"] = {"g", "ou"};
    m_pinyin_dict["gu"] = {"g", "u"};
    m_pinyin_dict["gua"] = {"g", "ua"};
    m_pinyin_dict["guai"] = {"g", "uai"};
    m_pinyin_dict["guan"] = {"g", "uan"};
    m_pinyin_dict["guang"] = {"g", "uang"};
    m_pinyin_dict["gui"] = {"g", "uei"};
    m_pinyin_dict["gun"] = {"g", "uen"};
    m_pinyin_dict["guo"] = {"g", "uo"};
    m_pinyin_dict["ha"] = {"h", "a"};
    m_pinyin_dict["hai"] = {"h", "ai"};
    m_pinyin_dict["han"] = {"h", "an"};
    m_pinyin_dict["hang"] = {"h", "ang"};
    m_pinyin_dict["hao"] = {"h", "ao"};
    m_pinyin_dict["he"] = {"h", "e"};
    m_pinyin_dict["hei"] = {"h", "ei"};
    m_pinyin_dict["hen"] = {"h", "en"};
    m_pinyin_dict["heng"] = {"h", "eng"};
    m_pinyin_dict["hong"] = {"h", "ong"};
    m_pinyin_dict["hou"] = {"h", "ou"};
    m_pinyin_dict["hu"] = {"h", "u"};
    m_pinyin_dict["hua"] = {"h", "ua"};
    m_pinyin_dict["huai"] = {"h", "uai"};
    m_pinyin_dict["huan"] = {"h", "uan"};
    m_pinyin_dict["huang"] = {"h", "uang"};
    m_pinyin_dict["hui"] = {"h", "uei"};
    m_pinyin_dict["hun"] = {"h", "uen"};
    m_pinyin_dict["huo"] = {"h", "uo"};
    m_pinyin_dict["ji"] = {"j", "i"};
    m_pinyin_dict["jia"] = {"j", "ia"};
    m_pinyin_dict["jian"] = {"j", "ian"};
    m_pinyin_dict["jiang"] = {"j", "iang"};
    m_pinyin_dict["jiao"] = {"j", "iao"};
    m_pinyin_dict["jie"] = {"j", "ie"};
    m_pinyin_dict["jin"] = {"j", "in"};
    m_pinyin_dict["jing"] = {"j", "ing"};
    m_pinyin_dict["jiong"] = {"j", "iong"};
    m_pinyin_dict["jiu"] = {"j", "iou"};
    m_pinyin_dict["ju"] = {"j", "v"};
    m_pinyin_dict["juan"] = {"j", "van"};
    m_pinyin_dict["jue"] = {"j", "ve"};
    m_pinyin_dict["jun"] = {"j", "vn"};
    m_pinyin_dict["ka"] = {"k", "a"};
    m_pinyin_dict["kai"] = {"k", "ai"};
    m_pinyin_dict["kan"] = {"k", "an"};
    m_pinyin_dict["kang"] = {"k", "ang"};
    m_pinyin_dict["kao"] = {"k", "ao"};
    m_pinyin_dict["ke"] = {"k", "e"};
    m_pinyin_dict["kei"] = {"k", "ei"};
    m_pinyin_dict["ken"] = {"k", "en"};
    m_pinyin_dict["keng"] = {"k", "eng"};
    m_pinyin_dict["kong"] = {"k", "ong"};
    m_pinyin_dict["kou"] = {"k", "ou"};
    m_pinyin_dict["ku"] = {"k", "u"};
    m_pinyin_dict["kua"] = {"k", "ua"};
    m_pinyin_dict["kuai"] = {"k", "uai"};
    m_pinyin_dict["kuan"] = {"k", "uan"};
    m_pinyin_dict["kuang"] = {"k", "uang"};
    m_pinyin_dict["kui"] = {"k", "uei"};
    m_pinyin_dict["kun"] = {"k", "uen"};
    m_pinyin_dict["kuo"] = {"k", "uo"};
    m_pinyin_dict["la"] = {"l", "a"};
    m_pinyin_dict["lai"] = {"l", "ai"};
    m_pinyin_dict["lan"] = {"l", "an"};
    m_pinyin_dict["lang"] = {"l", "ang"};
    m_pinyin_dict["lao"] = {"l", "ao"};
    m_pinyin_dict["le"] = {"l", "e"};
    m_pinyin_dict["lei"] = {"l", "ei"};
    m_pinyin_dict["leng"] = {"l", "eng"};
    m_pinyin_dict["li"] = {"l", "i"};
    m_pinyin_dict["lia"] = {"l", "ia"};
    m_pinyin_dict["lian"] = {"l", "ian"};
    m_pinyin_dict["liang"] = {"l", "iang"};
    m_pinyin_dict["liao"] = {"l", "iao"};
    m_pinyin_dict["lie"] = {"l", "ie"};
    m_pinyin_dict["lin"] = {"l", "in"};
    m_pinyin_dict["ling"] = {"l", "ing"};
    m_pinyin_dict["liu"] = {"l", "iou"};
    m_pinyin_dict["lo"] = {"l", "o"};
    m_pinyin_dict["long"] = {"l", "ong"};
    m_pinyin_dict["lou"] = {"l", "ou"};
    m_pinyin_dict["lu"] = {"l", "u"};
    m_pinyin_dict["lv"] = {"l", "v"};
    m_pinyin_dict["luan"] = {"l", "uan"};
    m_pinyin_dict["lve"] = {"l", "ve"};
    m_pinyin_dict["lue"] = {"l", "ve"};
    m_pinyin_dict["lun"] = {"l", "uen"};
    m_pinyin_dict["luo"] = {"l", "uo"};
    m_pinyin_dict["ma"] = {"m", "a"};
    m_pinyin_dict["mai"] = {"m", "ai"};
    m_pinyin_dict["man"] = {"m", "an"};
    m_pinyin_dict["mang"] = {"m", "ang"};
    m_pinyin_dict["mao"] = {"m", "ao"};
    m_pinyin_dict["me"] = {"m", "e"};
    m_pinyin_dict["mei"] = {"m", "ei"};
    m_pinyin_dict["men"] = {"m", "en"};
    m_pinyin_dict["meng"] = {"m", "eng"};
    m_pinyin_dict["mi"] = {"m", "i"};
    m_pinyin_dict["mian"] = {"m", "ian"};
    m_pinyin_dict["miao"] = {"m", "iao"};
    m_pinyin_dict["mie"] = {"m", "ie"};
    m_pinyin_dict["min"] = {"m", "in"};
    m_pinyin_dict["ming"] = {"m", "ing"};
    m_pinyin_dict["miu"] = {"m", "iou"};
    m_pinyin_dict["mo"] = {"m", "o"};
    m_pinyin_dict["mou"] = {"m", "ou"};
    m_pinyin_dict["mu"] = {"m", "u"};
    m_pinyin_dict["n"] = {"^", "en"};
    m_pinyin_dict["na"] = {"n", "a"};
    m_pinyin_dict["nai"] = {"n", "ai"};
    m_pinyin_dict["nan"] = {"n", "an"};
    m_pinyin_dict["nang"] = {"n", "ang"};
    m_pinyin_dict["nao"] = {"n", "ao"};
    m_pinyin_dict["ne"] = {"n", "e"};
    m_pinyin_dict["nei"] = {"n", "ei"};
    m_pinyin_dict["nen"] = {"n", "en"};
    m_pinyin_dict["neng"] = {"n", "eng"};
    m_pinyin_dict["ni"] = {"n", "i"};
    m_pinyin_dict["nia"] = {"n", "ia"};
    m_pinyin_dict["nian"] = {"n", "ian"};
    m_pinyin_dict["niang"] = {"n", "iang"};
    m_pinyin_dict["niao"] = {"n", "iao"};
    m_pinyin_dict["nie"] = {"n", "ie"};
    m_pinyin_dict["nin"] = {"n", "in"};
    m_pinyin_dict["ning"] = {"n", "ing"};
    m_pinyin_dict["niu"] = {"n", "iou"};
    m_pinyin_dict["nong"] = {"n", "ong"};
    m_pinyin_dict["nou"] = {"n", "ou"};
    m_pinyin_dict["nu"] = {"n", "u"};
    m_pinyin_dict["nv"] = {"n", "v"};
    m_pinyin_dict["nuan"] = {"n", "uan"};
    m_pinyin_dict["nve"] = {"n", "ve"};
    m_pinyin_dict["nue"] = {"n", "ve"};
    m_pinyin_dict["nuo"] = {"n", "uo"};
    m_pinyin_dict["o"] = {"^", "o"};
    m_pinyin_dict["ou"] = {"^", "ou"};
    m_pinyin_dict["pa"] = {"p", "a"};
    m_pinyin_dict["pai"] = {"p", "ai"};
    m_pinyin_dict["pan"] = {"p", "an"};
    m_pinyin_dict["pang"] = {"p", "ang"};
    m_pinyin_dict["pao"] = {"p", "ao"};
    m_pinyin_dict["pe"] = {"p", "e"};
    m_pinyin_dict["pei"] = {"p", "ei"};
    m_pinyin_dict["pen"] = {"p", "en"};
    m_pinyin_dict["peng"] = {"p", "eng"};
    m_pinyin_dict["pi"] = {"p", "i"};
    m_pinyin_dict["pian"] = {"p", "ian"};
    m_pinyin_dict["piao"] = {"p", "iao"};
    m_pinyin_dict["pie"] = {"p", "ie"};
    m_pinyin_dict["pin"] = {"p", "in"};
    m_pinyin_dict["ping"] = {"p", "ing"};
    m_pinyin_dict["po"] = {"p", "o"};
    m_pinyin_dict["pou"] = {"p", "ou"};
    m_pinyin_dict["pu"] = {"p", "u"};
    m_pinyin_dict["qi"] = {"q", "i"};
    m_pinyin_dict["qia"] = {"q", "ia"};
    m_pinyin_dict["qian"] = {"q", "ian"};
    m_pinyin_dict["qiang"] = {"q", "iang"};
    m_pinyin_dict["qiao"] = {"q", "iao"};
    m_pinyin_dict["qie"] = {"q", "ie"};
    m_pinyin_dict["qin"] = {"q", "in"};
    m_pinyin_dict["qing"] = {"q", "ing"};
    m_pinyin_dict["qiong"] = {"q", "iong"};
    m_pinyin_dict["qiu"] = {"q", "iou"};
    m_pinyin_dict["qu"] = {"q", "v"};
    m_pinyin_dict["quan"] = {"q", "van"};
    m_pinyin_dict["que"] = {"q", "ve"};
    m_pinyin_dict["qun"] = {"q", "vn"};
    m_pinyin_dict["ran"] = {"r", "an"};
    m_pinyin_dict["rang"] = {"r", "ang"};
    m_pinyin_dict["rao"] = {"r", "ao"};
    m_pinyin_dict["re"] = {"r", "e"};
    m_pinyin_dict["ren"] = {"r", "en"};
    m_pinyin_dict["reng"] = {"r", "eng"};
    m_pinyin_dict["ri"] = {"r", "iii"};
    m_pinyin_dict["rong"] = {"r", "ong"};
    m_pinyin_dict["rou"] = {"r", "ou"};
    m_pinyin_dict["ru"] = {"r", "u"};
    m_pinyin_dict["rua"] = {"r", "ua"};
    m_pinyin_dict["ruan"] = {"r", "uan"};
    m_pinyin_dict["rui"] = {"r", "uei"};
    m_pinyin_dict["run"] = {"r", "uen"};
    m_pinyin_dict["ruo"] = {"r", "uo"};
    m_pinyin_dict["sa"] = {"s", "a"};
    m_pinyin_dict["sai"] = {"s", "ai"};
    m_pinyin_dict["san"] = {"s", "an"};
    m_pinyin_dict["sang"] = {"s", "ang"};
    m_pinyin_dict["sao"] = {"s", "ao"};
    m_pinyin_dict["se"] = {"s", "e"};
    m_pinyin_dict["sen"] = {"s", "en"};
    m_pinyin_dict["seng"] = {"s", "eng"};
    m_pinyin_dict["sha"] = {"sh", "a"};
    m_pinyin_dict["shai"] = {"sh", "ai"};
    m_pinyin_dict["shan"] = {"sh", "an"};
    m_pinyin_dict["shang"] = {"sh", "ang"};
    m_pinyin_dict["shao"] = {"sh", "ao"};
    m_pinyin_dict["she"] = {"sh", "e"};
    m_pinyin_dict["shei"] = {"sh", "ei"};
    m_pinyin_dict["shen"] = {"sh", "en"};
    m_pinyin_dict["sheng"] = {"sh", "eng"};
    m_pinyin_dict["shi"] = {"sh", "iii"};
    m_pinyin_dict["shou"] = {"sh", "ou"};
    m_pinyin_dict["shu"] = {"sh", "u"};
    m_pinyin_dict["shua"] = {"sh", "ua"};
    m_pinyin_dict["shuai"] = {"sh", "uai"};
    m_pinyin_dict["shuan"] = {"sh", "uan"};
    m_pinyin_dict["shuang"] = {"sh", "uang"};
    m_pinyin_dict["shui"] = {"sh", "uei"};
    m_pinyin_dict["shun"] = {"sh", "uen"};
    m_pinyin_dict["shuo"] = {"sh", "uo"};
    m_pinyin_dict["si"] = {"s", "ii"};
    m_pinyin_dict["song"] = {"s", "ong"};
    m_pinyin_dict["sou"] = {"s", "ou"};
    m_pinyin_dict["su"] = {"s", "u"};
    m_pinyin_dict["suan"] = {"s", "uan"};
    m_pinyin_dict["sui"] = {"s", "uei"};
    m_pinyin_dict["sun"] = {"s", "uen"};
    m_pinyin_dict["suo"] = {"s", "uo"};
    m_pinyin_dict["ta"] = {"t", "a"};
    m_pinyin_dict["tai"] = {"t", "ai"};
    m_pinyin_dict["tan"] = {"t", "an"};
    m_pinyin_dict["tang"] = {"t", "ang"};
    m_pinyin_dict["tao"] = {"t", "ao"};
    m_pinyin_dict["te"] = {"t", "e"};
    m_pinyin_dict["tei"] = {"t", "ei"};
    m_pinyin_dict["teng"] = {"t", "eng"};
    m_pinyin_dict["ti"] = {"t", "i"};
    m_pinyin_dict["tian"] = {"t", "ian"};
    m_pinyin_dict["tiao"] = {"t", "iao"};
    m_pinyin_dict["tie"] = {"t", "ie"};
    m_pinyin_dict["ting"] = {"t", "ing"};
    m_pinyin_dict["tong"] = {"t", "ong"};
    m_pinyin_dict["tou"] = {"t", "ou"};
    m_pinyin_dict["tu"] = {"t", "u"};
    m_pinyin_dict["tuan"] = {"t", "uan"};
    m_pinyin_dict["tui"] = {"t", "uei"};
    m_pinyin_dict["tun"] = {"t", "uen"};
    m_pinyin_dict["tuo"] = {"t", "uo"};
    m_pinyin_dict["wa"] = {"^", "ua"};
    m_pinyin_dict["wai"] = {"^", "uai"};
    m_pinyin_dict["wan"] = {"^", "uan"};
    m_pinyin_dict["wang"] = {"^", "uang"};
    m_pinyin_dict["wei"] = {"^", "uei"};
    m_pinyin_dict["wen"] = {"^", "uen"};
    m_pinyin_dict["weng"] = {"^", "ueng"};
    m_pinyin_dict["wo"] = {"^", "uo"};
    m_pinyin_dict["wu"] = {"^", "u"};
    m_pinyin_dict["xi"] = {"x", "i"};
    m_pinyin_dict["xia"] = {"x", "ia"};
    m_pinyin_dict["xian"] = {"x", "ian"};
    m_pinyin_dict["xiang"] = {"x", "iang"};
    m_pinyin_dict["xiao"] = {"x", "iao"};
    m_pinyin_dict["xie"] = {"x", "ie"};
    m_pinyin_dict["xin"] = {"x", "in"};
    m_pinyin_dict["xing"] = {"x", "ing"};
    m_pinyin_dict["xiong"] = {"x", "iong"};
    m_pinyin_dict["xiu"] = {"x", "iou"};
    m_pinyin_dict["xu"] = {"x", "v"};
    m_pinyin_dict["xuan"] = {"x", "van"};
    m_pinyin_dict["xue"] = {"x", "ve"};
    m_pinyin_dict["xun"] = {"x", "vn"};
    m_pinyin_dict["ya"] = {"^", "ia"};
    m_pinyin_dict["yan"] = {"^", "ian"};
    m_pinyin_dict["yang"] = {"^", "iang"};
    m_pinyin_dict["yao"] = {"^", "iao"};
    m_pinyin_dict["ye"] = {"^", "ie"};
    m_pinyin_dict["yi"] = {"^", "i"};
    m_pinyin_dict["yin"] = {"^", "in"};
    m_pinyin_dict["ying"] = {"^", "ing"};
    m_pinyin_dict["yo"] = {"^", "iou"};
    m_pinyin_dict["yong"] = {"^", "iong"};
    m_pinyin_dict["you"] = {"^", "iou"};
    m_pinyin_dict["yu"] = {"^", "v"};
    m_pinyin_dict["yuan"] = {"^", "van"};
    m_pinyin_dict["yue"] = {"^", "ve"};
    m_pinyin_dict["yun"] = {"^", "vn"};
    m_pinyin_dict["za"] = {"z", "a"};
    m_pinyin_dict["zai"] = {"z", "ai"};
    m_pinyin_dict["zan"] = {"z", "an"};
    m_pinyin_dict["zang"] = {"z", "ang"};
    m_pinyin_dict["zao"] = {"z", "ao"};
    m_pinyin_dict["ze"] = {"z", "e"};
    m_pinyin_dict["zei"] = {"z", "ei"};
    m_pinyin_dict["zen"] = {"z", "en"};
    m_pinyin_dict["zeng"] = {"z", "eng"};
    m_pinyin_dict["zha"] = {"zh", "a"};
    m_pinyin_dict["zhai"] = {"zh", "ai"};
    m_pinyin_dict["zhan"] = {"zh", "an"};
    m_pinyin_dict["zhang"] = {"zh", "ang"};
    m_pinyin_dict["zhao"] = {"zh", "ao"};
    m_pinyin_dict["zhe"] = {"zh", "e"};
    m_pinyin_dict["zhei"] = {"zh", "ei"};
    m_pinyin_dict["zhen"] = {"zh", "en"};
    m_pinyin_dict["zheng"] = {"zh", "eng"};
    m_pinyin_dict["zhi"] = {"zh", "iii"};
    m_pinyin_dict["zhong"] = {"zh", "ong"};
    m_pinyin_dict["zhou"] = {"zh", "ou"};
    m_pinyin_dict["zhu"] = {"zh", "u"};
    m_pinyin_dict["zhua"] = {"zh", "ua"};
    m_pinyin_dict["zhuai"] = {"zh", "uai"};
    m_pinyin_dict["zhuan"] = {"zh", "uan"};
    m_pinyin_dict["zhuang"] = {"zh", "uang"};
    m_pinyin_dict["zhui"] = {"zh", "uei"};
    m_pinyin_dict["zhun"] = {"zh", "uen"};
    m_pinyin_dict["zhuo"] = {"zh", "uo"};
    m_pinyin_dict["zi"] = {"z", "ii"};
    m_pinyin_dict["zong"] = {"z", "ong"};
    m_pinyin_dict["zou"] = {"z", "ou"};
    m_pinyin_dict["zu"] = {"z", "u"};
    m_pinyin_dict["zuan"] = {"z", "uan"};
    m_pinyin_dict["zui"] = {"z", "uei"};
    m_pinyin_dict["zun"] = {"z", "uen"};
    m_pinyin_dict["zuo"] = {"z", "uo"};
}

void VitsPreprocessor::build_symbol_table() {
    vector<string> pauses = {"sil", "eos", "sp", "#0", "#1", "#2", "#3"};
    vector<string> initials = {
        "^", "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m",
        "n", "p", "q", "r", "s", "sh", "t", "x", "z", "zh"
    };
    vector<string> finals = {
        "a", "ai", "an", "ang", "ao", "e", "ei", "en", "eng", "er",
        "i", "ia", "ian", "iang", "iao", "ie", "ii", "iii", "in", "ing",
        "iong", "iou", "o", "ong", "ou", "u", "ua", "uai", "uan", "uang",
        "uei", "uen", "ueng", "uo", "v", "van", "ve", "vn"
    };
    vector<string> tones = {"1", "2", "3", "4", "5"};

    int id = 0;
    for (const auto& s : pauses) m_symbol_to_id[s] = id++;
    for (const auto& s : initials) m_symbol_to_id[s] = id++;
    for (const auto& f : finals) {
        for (const auto& t : tones) {
            m_symbol_to_id[f + t] = id++;
        }
    }
    cout << "Built symbol table: " << m_symbol_to_id.size() << " symbols" << endl;
}

int VitsPreprocessor::process(const string& text,
                                vector<int32_t>& input_ids,
                                vector<float>& char_embeds) {
    m_ts->save("preprocess");

    // 1. Clean text
    string cleaned = clean_chinese(text);

    // 2. Build phonemes and char list
    vector<string> phonemes = {"sil"};
    vector<string> chars = {"[PAD]"};
    vector<int> count_phone = {1};

    string subtext;
    for (size_t i = 0; i < cleaned.size();) {
        uint32_t cp;
        int len = utf8_decode(cleaned, i, cp);
        string ch = cleaned.substr(i, len);
        i += len;

        if (ch == ",") {
            if (subtext.empty()) continue;
            // Process accumulated subtext
            vector<string> pinyins;
            for (size_t j = 0; j < subtext.size();) {
                uint32_t scp;
                int slen = utf8_decode(subtext, j, scp);
                string sch = subtext.substr(j, slen);
                j += slen;
                string py = char_to_pinyin(sch);
                if (!py.empty()) pinyins.push_back(py);
            }
            vector<int> sub_c;
            auto sub_p = get_phoneme4pinyin(pinyins, sub_c);
            phonemes.insert(phonemes.end(), sub_p.begin(), sub_p.end());
            phonemes.push_back("sp");
            count_phone.insert(count_phone.end(), sub_c.begin(), sub_c.end());
            count_phone.push_back(1);
            chars.push_back(subtext);
            chars.push_back(",");
            subtext.clear();
        } else if (is_chinese(cp)) {
            subtext += ch;
        }
    }
    // Last subtext
    if (!subtext.empty()) {
        vector<string> pinyins;
        for (size_t j = 0; j < subtext.size();) {
            uint32_t scp;
            int slen = utf8_decode(subtext, j, scp);
            string sch = subtext.substr(j, slen);
            j += slen;
            string py = char_to_pinyin(sch);
            if (!py.empty()) pinyins.push_back(py);
        }
        vector<int> sub_c;
        auto sub_p = get_phoneme4pinyin(pinyins, sub_c);
        phonemes.insert(phonemes.end(), sub_p.begin(), sub_p.end());
        phonemes.push_back("sp");
        count_phone.insert(count_phone.end(), sub_c.begin(), sub_c.end());
        count_phone.push_back(1);
        chars.push_back(subtext);
        chars.push_back(",");
    }

    phonemes.push_back("sil");
    count_phone.push_back(1);
    chars.push_back("[PAD]");

    // 3. Convert phonemes to IDs
    input_ids.clear();
    for (const auto& p : phonemes) {
        auto it = m_symbol_to_id.find(p);
        if (it != m_symbol_to_id.end()) {
            input_ids.push_back(it->second);
        }
    }
    m_ts->save("preprocess.phonemes");

    // 4. BERT inference to get char_embeds
    string chars_str;
    for (const auto& c : chars) chars_str += c;

    // Tokenize: each Chinese char (or group) + punctuation
    vector<int> token_ids;
    // [CLS] token is typically [PAD]'s id in this vocab, but let's just use the vocab
    for (size_t i = 0; i < chars_str.size();) {
        // Try to find the longest match
        bool found = false;
        for (size_t j = i + 1; j <= chars_str.size(); j++) {
            string sub = chars_str.substr(i, j - i);
            auto it = m_vocab.find(sub);
            if (it != m_vocab.end()) {
                token_ids.push_back(it->second);
                i = j;
                found = true;
                break;
            }
        }
        if (!found) {
            // Try single char
            string single = chars_str.substr(i, 1);
            auto it = m_vocab.find(single);
            if (it != m_vocab.end()) {
                token_ids.push_back(it->second);
            } else {
                // Use [UNK] if available, otherwise skip
                auto unk = m_vocab.find("[UNK]");
                if (unk != m_vocab.end()) token_ids.push_back(unk->second);
            }
            i++;
        }
    }

    int seq_len = (int)token_ids.size();
    if (seq_len > m_max_length) seq_len = m_max_length;

    // Create BERT input tensors
    bm_tensor_t input_tensors[3];
    bm_tensor_t output_tensor;

    bm_shape_t shape;
    shape.num_dims = 2;
    shape.dims[0] = 1;
    shape.dims[1] = m_max_length;

    // Create input tensors (BERT uses FLOAT32 inputs)
    bool ok = bmrt_tensor(&input_tensors[0], bmrt, BM_FLOAT32, shape);
    assert(ok == true);
    ok = bmrt_tensor(&input_tensors[1], bmrt, BM_FLOAT32, shape);
    assert(ok == true);
    ok = bmrt_tensor(&input_tensors[2], bmrt, BM_FLOAT32, shape);
    assert(ok == true);

    // Fill data (convert to float32)
    vector<float> padded_ids(m_max_length, 0.0f);
    vector<float> padded_masks(m_max_length, 0.0f);
    vector<float> padded_types(m_max_length, 0.0f);

    for (int i = 0; i < seq_len; i++) {
        padded_ids[i] = static_cast<float>(token_ids[i]);
        padded_masks[i] = 1.0f;
        padded_types[i] = 0.0f;
    }

    bm_memcpy_s2d(handle, input_tensors[0].device_mem, padded_ids.data());
    bm_memcpy_s2d(handle, input_tensors[1].device_mem, padded_masks.data());
    bm_memcpy_s2d(handle, input_tensors[2].device_mem, padded_types.data());

    // Launch BERT
    ok = bmrt_launch_tensor(bmrt, netinfo->name, input_tensors, netinfo->input_num,
                            &output_tensor, netinfo->output_num);
    assert(ok == true);
    bm_thread_sync(handle);

    // Free input device mem
    for (int i = 0; i < 3; i++) {
        bm_free_device(handle, input_tensors[i].device_mem);
    }

    // Get output
    float* bert_output = get_cpu_data(&output_tensor);
    int output_size = output_tensor.shape.dims[1] * output_tensor.shape.dims[2]; // max_length * embed_dim

    m_ts->save("preprocess.bert");

    // 5. Expand for phone
    vector<float> bert_vec(bert_output, bert_output + output_size);
    bm_free_device(handle, output_tensor.device_mem);
    delete [] bert_output;

    // Expand: for each char embedding, repeat by count_phone[j]
    int expand_len = m_max_length * 2; // = 128
    int num_chars = chars.size();
    vector<float> expand_embeds(expand_len * m_embed_dim, 0.0f);

    int dst_offset = 0;
    for (int j = 0; j < num_chars && j < m_max_length && dst_offset / m_embed_dim < expand_len; j++) {
        int repeat = (j < (int)count_phone.size()) ? count_phone[j] : 1;
        for (int r = 0; r < repeat && dst_offset / m_embed_dim < expand_len; r++) {
            memcpy(&expand_embeds[dst_offset],
                   &bert_vec[j * m_embed_dim],
                   m_embed_dim * sizeof(float));
            dst_offset += m_embed_dim;
        }
    }

    char_embeds = std::move(expand_embeds);

    m_ts->save("preprocess");
    return 0;
}