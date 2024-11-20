#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <utf8proc.h>
#include <sstream>
#include <codecvt>

//https://unicode.org/reports/tr15/#Norm_Forms
//https://ssl.icu-project.org/apiref/icu4c/uchar_8h.html

std::filesystem::path script_path = std::filesystem::current_path();
std::string vocab_path = (script_path / "tokenizer/vocab.txt").string();

// std::string current_file_name = __FILE__;
// std::string current_file_path = current_file_name.substr(0,current_file_name.find_last_of("/"));
// std::string vocab_path = current_file_path + "/vocab.txt";

const std::wstring stripChar = L" \t\n\r\v\f";
using Vocab = std::unordered_map<std::wstring, size_t>;
using InvVocab = std::unordered_map<size_t, std::wstring>;

std::vector<std::wstring> wsplit(const std::wstring &ws, const std::wstring &delimiters) {
    std::vector<std::wstring> tokens;
    size_t start = 0, end = 0;

    while ((start = ws.find_first_not_of(delimiters, end)) != std::wstring::npos) {
        end = ws.find_first_of(delimiters, start);
        tokens.push_back(ws.substr(start, end - start));
    }

    return tokens;
}

std::wstring wjoin(const std::vector<std::wstring> &tokens, const std::wstring &delimiter) {
    std::wostringstream oss;
    bool first = true;

    for (const auto &token : tokens) {
        if (!first) {
            oss << delimiter;
        } else {
            first = false;
        }

        oss << token;
    }

    return oss.str();
}


class BasicTokenizer {
public:
    BasicTokenizer(bool doLowerCase = true);
    std::vector<std::wstring> tokenize(const std::string& text) const;

private:
    std::wstring cleanText(const std::wstring& text) const;
    bool isControol(const wchar_t& ch) const;
    bool isWhitespace(const wchar_t& ch) const;
    bool isPunctuation(const wchar_t& ch) const;
    bool isChineseChar(const wchar_t& ch) const;
    std::wstring tokenizeChineseChars(const std::wstring& text) const;
    bool isStripChar(const wchar_t& ch) const;
    std::wstring strip(const std::wstring& text) const;
    std::vector<std::wstring> split(const std::wstring& text) const;
    std::wstring runStripAccents(const std::wstring& text) const;
    std::vector<std::wstring> runSplitOnPunc(const std::wstring& text) const;

    bool mDoLowerCase;
};

class WordpieceTokenizer {
public:
    WordpieceTokenizer(std::shared_ptr<Vocab> vocab, const std::wstring& unkToken = L"[UNK]", size_t maxInputCharsPerWord=200);
    std::vector<std::wstring> tokenize(const std::wstring& text) const;

private:
    std::shared_ptr<Vocab> mVocab;
    std::wstring mUnkToken;
    size_t mMaxInputCharsPerWord;
};


class FullTokenizer {
public:
    FullTokenizer(const std::string vocabFile = vocab_path, bool doLowerCase = true);
    std::vector<std::wstring> tokenize(const std::string& text) const;
    std::vector<int> convertTokensToIds(const std::vector<std::wstring>& text) const;

    std::vector<int> encode(const std::string& text, int max_length, bool add_special_token = true,
                        bool padding = true) const;
    std::string decode(std::vector<size_t>& input_id);


private:
    std::shared_ptr<Vocab> mVocab;
    InvVocab mInvVocab;
    std::string mVocabFile;
    bool mDoLowerCase;
    BasicTokenizer mBasicTokenizer;
    WordpieceTokenizer mWordpieceTokenizer;

    const int BOS_TOKEN_ID = 101;
    const int EOS_TOKEN_ID = 102;
    const int PAD_TOKEN_ID = 0;
};
using BertTokenizer = FullTokenizer;

static std::string normalize_nfd(const std::string& s) {
    std::string ret;
    char *result = (char *) utf8proc_NFD((unsigned char *)s.c_str());
    if (result) {
        ret = std::string(result);
        free(result);
        result = NULL;
    } 
    return ret;
}

static bool isStripChar(const wchar_t& ch) {
    return stripChar.find(ch) != std::wstring::npos;
}

static std::wstring strip(const std::wstring& text) {
    std::wstring ret =  text;
    if (ret.empty()) return ret;
    size_t pos = 0;
    while (pos < ret.size() && isStripChar(ret[pos])) pos++;
    if (pos != 0) ret = ret.substr(pos, ret.size() - pos);
    pos = ret.size() - 1;
    while (pos != (size_t)-1 && isStripChar(ret[pos])) pos--;
    return ret.substr(0, pos + 1);
}

static std::vector<std::wstring> split(const std::wstring& text) {
    std::vector<std::wstring>  result = wsplit(text, stripChar);
    return result;
}

static std::vector<std::wstring> whitespaceTokenize(const std::wstring& text) {
    std::wstring rtext = strip(text);
    if (rtext.empty()) return std::vector<std::wstring>();
    return split(text);
}

static std::wstring convertToUnicode(const std::string& text) {
    size_t i = 0;
    std::wstring ret;
    while (i < text.size()) {
        wchar_t codepoint;
        utf8proc_ssize_t forward = utf8proc_iterate((utf8proc_uint8_t *)&text[i], text.size() - i, (utf8proc_int32_t*)&codepoint);
        if (forward < 0) return L"";
        ret += codepoint;
        i += forward;
    }
    return ret;
}

static std::string convertFromUnicode(const std::wstring& wText) {
    char dst[64];
    std::string ret;
    for (auto ch : wText) {
        utf8proc_ssize_t num = utf8proc_encode_char(ch, (utf8proc_uint8_t *)dst);
        if (num <= 0) return "";
        ret += std::string(dst, dst+num);
    }
    return ret;
}

static std::wstring tolower(const std::wstring& s) {
    std::wstring ret(s.size(), L' ');
    for (size_t i = 0; i < s.size(); i++) {
        ret[i] = utf8proc_tolower(s[i]);
    }
    return ret;
}

static std::shared_ptr<Vocab> loadVocab(const std::string& vocabFile) {
    std::shared_ptr<Vocab> vocab(new Vocab);
    size_t index = 0;
    std::ifstream ifs(vocabFile, std::ifstream::in);
    std::string line;
    while (getline(ifs, line)) {
        std::wstring token = convertToUnicode(line);
        if (token.empty()) break;
        token = strip(token);
        (*vocab)[token] = index;
        index++;
    }
    return vocab;
}

BasicTokenizer::BasicTokenizer(bool doLowerCase) 
    : mDoLowerCase(doLowerCase) {
}

std::wstring BasicTokenizer::cleanText(const std::wstring& text) const {
    std::wstring output;
    for (const wchar_t& cp : text)  {
        if (cp == 0 || cp == 0xfffd || isControol(cp)) continue;
        if (isWhitespace(cp)) output += L" ";
        else output += cp;
    }
    return output;
}

bool BasicTokenizer::isControol(const wchar_t& ch) const {
    if (ch== L'\t' || ch== L'\n' || ch== L'\r') return false;
    auto cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF) return true;
    return false;
}

bool BasicTokenizer::isWhitespace(const wchar_t& ch) const {
    if (ch== L' ' || ch== L'\t' || ch== L'\n' || ch== L'\r') return true;
    auto cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_ZS) return true;
    return false;
}

bool BasicTokenizer::isPunctuation(const wchar_t& ch) const {
    if ((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) || 
        (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126)) return true;
    auto cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS
            || cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC
            || cat == UTF8PROC_CATEGORY_PO //sometimes ¶ belong SO
            || cat == UTF8PROC_CATEGORY_PI
            || cat == UTF8PROC_CATEGORY_PF) return true;
    return false;
}

bool BasicTokenizer::isChineseChar(const wchar_t& ch) const {
    if ((ch >= 0x4E00 && ch <= 0x9FFF) ||
        (ch >= 0x3400 && ch <= 0x4DBF) ||
        (ch >= 0x20000 && ch <= 0x2A6DF) ||  
        (ch >= 0x2A700 && ch <= 0x2B73F) ||
        (ch >= 0x2B740 && ch <= 0x2B81F) ||
        (ch >= 0x2B820 && ch <= 0x2CEAF) ||
        (ch >= 0xF900 && ch <= 0xFAFF) ||
        (ch >= 0x2F800 && ch <= 0x2FA1F))
        return true;
    return false;
}

std::wstring BasicTokenizer::tokenizeChineseChars(const std::wstring& text) const {
    std::wstring output;
    for (auto& ch : text) {
        if (isChineseChar(ch)) {
            output += L' ';
            output += ch;
            output += L' ';
        }
        else
            output += ch;
    }
    return output;
}

std::wstring BasicTokenizer::runStripAccents(const std::wstring& text) const {
    //Strips accents from a piece of text.
    std::wstring nText;
    try {
        nText = convertToUnicode(normalize_nfd(convertFromUnicode(text)));
    } catch (std::bad_cast& e) {
        std::cerr << "bad_cast" << std::endl;
        return L"";  
    }
    
    std::wstring output;
    for (auto& ch : nText) {
        auto cat = utf8proc_category(ch);
        if (cat == UTF8PROC_CATEGORY_MN) continue;
        output += ch;
    }
    return output;
}

std::vector<std::wstring> BasicTokenizer::runSplitOnPunc(const std::wstring& text) const {
    size_t i = 0;
    bool startNewWord = true;
    std::vector<std::wstring> output;
    while (i < text.size()) {
        wchar_t ch = text[i];
        if (isPunctuation(ch)) {
            output.push_back(std::wstring(&ch, 1));
            startNewWord = true;
        }
        else {
            if (startNewWord) output.push_back(std::wstring());
            startNewWord = false;
            output[output.size() - 1] += ch;
        }
        i++;
    }
    return output;
}

std::vector<std::wstring> BasicTokenizer::tokenize(const std::string& text) const {
    std::wstring nText = convertToUnicode(text);
    nText = cleanText(nText);

    nText = tokenizeChineseChars(nText);

    const std::vector<std::wstring>& origTokens = whitespaceTokenize(nText);
    std::vector<std::wstring> splitTokens;
    for (std::wstring token : origTokens) {
        if (mDoLowerCase) {
            token = tolower(token);
            token = runStripAccents(token);
        }
        const auto& tokens = runSplitOnPunc(token);
        splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
    }
    return whitespaceTokenize(wjoin(splitTokens, L" "));
}

WordpieceTokenizer::WordpieceTokenizer(const std::shared_ptr<Vocab> vocab, const std::wstring& unkToken, size_t maxInputCharsPerWord)
    : mVocab(vocab),
    mUnkToken(unkToken),
    mMaxInputCharsPerWord(maxInputCharsPerWord) {
}

std::vector<std::wstring> WordpieceTokenizer::tokenize(const std::wstring& text) const {
    std::vector<std::wstring> outputTokens;
    for (auto& token : whitespaceTokenize(text)) {
        if (token.size() > mMaxInputCharsPerWord) {
            outputTokens.push_back(mUnkToken);
        }
        bool isBad = false;
        size_t start = 0;
        std::vector<std::wstring> subTokens;
        while (start < token.size()) {
            size_t end = token.size();
            std::wstring curSubstr;
            bool hasCurSubstr = false;
            while (start < end) {
                std::wstring substr = token.substr(start, end - start);
                if (start > 0) substr = L"##" + substr;
                if (mVocab->find(substr) != mVocab->end()) {
                    curSubstr = substr;
                    hasCurSubstr = true;
                    break;
                }
                end--;
            }
            if (!hasCurSubstr) {
                isBad = true;
                break;
            }
            subTokens.push_back(curSubstr);
            start = end;
        }
        if (isBad) outputTokens.push_back(mUnkToken);
        else outputTokens.insert(outputTokens.end(), subTokens.begin(), subTokens.end());
    }
    return outputTokens;
}

FullTokenizer::FullTokenizer(const std::string vocabFile, bool doLowerCase) : 
    mVocab(loadVocab(vocabFile)), 
    mBasicTokenizer(BasicTokenizer(doLowerCase)),
    mWordpieceTokenizer(WordpieceTokenizer(mVocab)) {
    for (auto& v : *mVocab) mInvVocab[v.second] = v.first;
}

std::vector<int> FullTokenizer::encode(const std::string& text,int max_length, bool add_special_token,
    bool padding ) const{
    
    auto tokens = tokenize(text);
    std::vector<int> ids = convertTokensToIds(tokens);
    if(add_special_token){
        ids.insert(ids.begin(),BOS_TOKEN_ID);
        ids.insert(ids.end(),  EOS_TOKEN_ID);
    }
    if(padding){
        if(ids.size()<max_length){
            int n =ids.size();
            for(int i=0;i<max_length - n;i++){
                ids.push_back(0);
            }
        }
    }
    return ids;
}

std::string FullTokenizer::decode( std::vector<size_t>& input_ids){
    std::string s ="";
    for(int i=0;i<input_ids.size();i++){
        if (input_ids[i] == BOS_TOKEN_ID) continue;
        if(input_ids[i] == EOS_TOKEN_ID) break;
        std::wstring ws =  mInvVocab[input_ids[i]];
        s += convertFromUnicode(ws);
    }
    return s;    
}


std::vector<std::wstring> FullTokenizer::tokenize(const std::string& text) const {
    std::vector<std::wstring> splitTokens;
    for (auto& token : mBasicTokenizer.tokenize(text))
        for (auto& subToken : mWordpieceTokenizer.tokenize(token))  
            splitTokens.push_back(subToken);
    return splitTokens;
}

std::vector<int> FullTokenizer::convertTokensToIds(const std::vector<std::wstring>& text) const {
    std::vector<int> ret(text.size());
    for (size_t i = 0; i < text.size(); i++) {
        ret[i] = (*mVocab)[text[i]];
    }
    return ret;
}