//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include"SimpleTokenizer.hpp"

#include <fstream>
#include <sstream>
#include <filesystem>


SimpleTokenizer::SimpleTokenizer()
{
    this->Init();
}

SimpleTokenizer::~SimpleTokenizer()
{

}

// 初始化成员变量
void SimpleTokenizer::Init()
{
    this->byte_encoder = this->BytesToUnicode();

    // 将 byte_encoder 转换为 byte_decoder
    for (const auto& kvp : this->byte_encoder) {
        this->byte_decoder[kvp.second] = kvp.first;
    }

    std::filesystem::path script_path = std::filesystem::current_path();
    std::string bpePath = script_path.string() + "/bpe_simple_vocab_16e6.txt"; // 先转换为 std::string


    if (bpePath.empty()) {
        std::cerr << "Error: bpePath is empty!" << std::endl;
        return;
    }

    std::vector<std::tuple<std::wstring, std::wstring>> merges = LoadBPEMerges(bpePath); //加载BPE


    // 提取 byte_encoder 的值到 vocab
    std::vector<std::wstring> vocab;
    for (const auto& kvp : this->byte_encoder) {
        vocab.push_back(kvp.second);
    }
    for (const auto& kvp : this->byte_encoder) {
        vocab.push_back(kvp.second + L"</w>");
    }
    for (const auto& merge : merges) {
        vocab.push_back(std::get<0>(merge) + std::get<1>(merge));
    }

    vocab.push_back(L"<|startoftext|>");
    vocab.push_back(L"<|endoftext|>");


    for (int i = 0; i < vocab.size(); i++) {
        this->encoder[vocab[i]] = i;
    }

    for (const auto& kvp : encoder) {
        this->decoder[kvp.second] = kvp.first;
    }

    int index = 0;
    for (const auto& merge : merges) {
        this->bpe_ranks[merge] = index++;
    }

    this->cache[L"<|startoftext|>"] = L"<|startoftext|>";

    this->pat = std::wregex(L"(\s<\|startoftext\|\>|<\|endoftext\|\>|\s's|\s't|\s're|\s've|\s'm|\s'll|\s'd|[[:alpha:]]+|[[:digit:]]+|\S+)", std::regex_constants::icase);
}

std::vector<int64_t> SimpleTokenizer::tokenlize(std::wstring textpromot)
{
    int sot_token = this->encoder[L"<|startoftext|>"];
    int eot_token = this->encoder[L"<|endoftext|>"];

    std::vector<std::wstring>texts;
    texts.push_back(textpromot);

    std::vector<int64_t> allTokens;

    for (const std::wstring& text : texts)
    {
        allTokens.push_back(sot_token);
        std::vector<int64_t> encodedText = this->Encode(text);
        allTokens.insert(allTokens.end(), encodedText.begin(), encodedText.end());
        allTokens.push_back(eot_token);
    }

    if (allTokens.size() > contextLength)
    {
        allTokens.erase(allTokens.begin() + contextLength, allTokens.end());
        allTokens[contextLength - 1] = eot_token;
    }
    else
    {
        int addedCount = contextLength - allTokens.size();
        allTokens.insert(allTokens.end(), addedCount, 0);
    }
    return allTokens;
}
std::vector<int64_t> SimpleTokenizer::Encode(const std::wstring& text) {
    std::vector<int64_t> bpeTokens;
    std::wstring cleanedText = whitespace_clean(text);
    std::transform(cleanedText.begin(), cleanedText.end(), cleanedText.begin(), [](char c) {
        return std::tolower(static_cast<unsigned char>(c));
        });


    std::wsmatch match;
    std::wstring::const_iterator searchStart(cleanedText.cbegin());
    while (std::regex_search(searchStart, cleanedText.cend(), match, this->pat)) {
        std::wstring token = match[0].str();
        std::wstring encodedToken;
        for (wchar_t c : token) {
            encodedToken += this->byte_encoder[c];
        }

        std::vector<std::wstring> bpeTokenList = this->split(this->bpe(encodedToken), ' ');
        // Split encodedToken into bpeTokenList using ' ' delimiter
        // Implement splitting logic here
        for (const std::wstring& bpeToken : bpeTokenList) {
            bpeTokens.push_back(this->encoder[bpeToken]);
        }
        searchStart = match.suffix().first;
    }

    return bpeTokens;
}

std::vector<std::wstring> SimpleTokenizer::split(const std::wstring& str, wchar_t delimiter) {
    std::vector<std::wstring> tokens;
    std::wistringstream iss(str);
    std::wstring token;

    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

// 数字向量解码成文本
std::wstring SimpleTokenizer::Decode(const std::vector<int>& tokens) {
    std::wstringstream textBuilder;
    for (int token : tokens) {
        textBuilder << decoder[token];
    }
    std::wstring text = textBuilder.str();

    std::vector<uint8_t> byteList;
    for (char c : text) {
        byteList.push_back(static_cast<uint8_t>(byte_decoder[std::wstring(1, c)]));
    }
    std::wstring decodedText(byteList.begin(), byteList.end());
    std::replace(decodedText.begin(), decodedText.end(), '/', ' ');

    return decodedText;
}

std::tuple<std::wstring, std::wstring> SimpleTokenizer::FindFirstPair(const std::vector<std::pair<std::wstring, std::wstring>>& pairs, const std::unordered_map<std::tuple<std::wstring, std::wstring>, int, std::TupleHash>& bpe_ranks) {
    auto compareByRank = [&](const std::pair<std::wstring, std::wstring>& a, const std::pair<std::wstring, std::wstring>& b) {
        if (bpe_ranks.count(a) && bpe_ranks.count(b)) {
            return bpe_ranks.at(a) < bpe_ranks.at(b);
        }
        else if (bpe_ranks.count(a)) {
            return true;
        }
        else {
            return false;
        }
    };

    auto it = std::min_element(pairs.begin(), pairs.end(), compareByRank);

    return std::make_tuple(it->first, it->second);
}
std::wstring SimpleTokenizer::bpe(const std::wstring& token) {
    if (this->cache.count(token) > 0) {
        return cache[token];
    }

    std::vector<std::wstring> word;
    for (size_t i = 0; i < token.length() - 1; i++) {
        word.push_back(std::wstring(1, token[i]));
    }
    word.push_back(token.substr(token.length() - 1) + L"</w>");

    std::vector<std::pair<std::wstring, std::wstring>> pairs = GetPairs(word);

    if (pairs.empty()) {
        return token + L"</w>";
    }

    while (true) {
       /* auto minPair = std::min_element(pairs.begin(), pairs.end(),
            [&](const std::pair<std::wstring, std::wstring>& pair1, const std::pair<std::wstring, std::wstring>& pair2) {
                return bpe_ranks.count(pair1) ? bpe_ranks[pair1] : std::numeric_limits<double>::infinity() <
                    bpe_ranks.count(pair2) ? bpe_ranks[pair2] : std::numeric_limits<double>::infinity();
            });*/
        std::tuple<std::wstring, std::wstring> firstPair = FindFirstPair(pairs, this->bpe_ranks);
        
        if (bpe_ranks.find(firstPair) == bpe_ranks.end()) {
            break;
        }
        std::wstring first = std::get<0> (firstPair);
        std::wstring second = std::get<1>(firstPair);

        if (!bpe_ranks.count(std::make_pair(first, second))) {
            break;
        }

        std::vector<std::wstring> newWord;
        int i = 0;
        while (i < word.size()) {
            try {
                int j = std::find(word.begin() + i, word.end(), first) - word.begin();
                if (j == word.size())j = -1;

                if (j - i >= i)
                {
                    newWord.insert(newWord.end(), word.begin() + i, word.begin() + j);
                    i = j;
                }
                else
                {
                    newWord.insert(newWord.end(), word.begin() + i, word.end());
                    break;
                }
               
            }
            catch (...) {
                newWord.insert(newWord.end(), word.begin() + i, word.end());
                break;
            }

            if (word[i] == first && i < word.size() - 1 && word[i + 1] == second) {
                newWord.push_back(first + second);
                i += 2;
            }
            else {
                newWord.push_back(word[i]);
                i += 1;
            }
        }

        word = newWord;

        if (word.size() == 1) {
            break;
        }
        else {
            pairs = GetPairs(newWord);
        }
    }

    std::wstring result = L"";
    for (const std::wstring& w : word) {
        result += w + L" ";
    }
    result = result.substr(0, result.size() - 1); // Remove trailing space

    cache[token] = result;
    return result;
}

std::unordered_map<int, std::wstring> SimpleTokenizer::BytesToUnicode() {
    std::vector<int> bs;
    std::vector<int> cs;

    for (int b = static_cast<int>(L'!'); b <= static_cast<int>(L'~'); b++) {
        bs.push_back(b);
        cs.push_back(b);
    }

    for (int b = static_cast<int>(L'¡'); b <= static_cast<int>(L'¬'); b++) {
        bs.push_back(b);
        cs.push_back(b);
    }

    for (int b = static_cast<int>(L'®'); b <= static_cast<int>(L'ÿ'); b++) {
        bs.push_back(b);
        cs.push_back(b);
    }

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    std::unordered_map<int, std::wstring> byteToUnicode;
    for (size_t i = 0; i < bs.size(); i++) {
        byteToUnicode[bs[i]] = std::wstring(1, static_cast<wchar_t>(cs[i]));
    }

    return byteToUnicode;

}

std::vector<std::tuple<std::wstring, std::wstring>> SimpleTokenizer::LoadBPEMerges(const std::string& bpePath) {
    std::vector<std::tuple<std::wstring, std::wstring>> merges;

    // 使用 std::ifstream 打开文件
    std::ifstream fileStream(bpePath);
    if (!fileStream.is_open()) {
        std::wcout << L"File is not open." << std::endl;
        return merges;
    }
    std::wcout << L"File is open." << std::endl;

    std::string line;
    size_t startLine = 1;
    size_t endLine = 49152 - 256 - 2 + 1;
    std::vector<std::wstring> lines;

    // 逐行读取文件
    while (std::getline(fileStream, line)) {
        // 将 UTF-8 字符串转换为宽字符串
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::wstring wline = converter.from_bytes(line);
        
        if (wline.empty()) {
            continue; // 跳过空行
        }
        lines.push_back(wline);
    }

    // 创建行段
    std::vector<std::wstring> lineSegment(lines.begin() + startLine, lines.begin() + endLine);

    for (const std::wstring& line : lineSegment) {
        std::wistringstream streamReader(line);
        std::wstring merge[2];
        if (streamReader >> merge[0] >> merge[1]) { // 检查是否成功读取两个单词
            merges.emplace_back(merge[0], merge[1]);
        } else {
            std::wcerr << L"Failed to parse line: " << line << std::endl; // 解析失败
        }
    }

    fileStream.close();
    return merges;
}

std::vector<std::pair<std::wstring, std::wstring>> SimpleTokenizer::GetPairs(const std::vector<std::wstring>& words) {
    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    
    std::wstring prevChar = words[0];

    for (size_t i = 1; i < words.size(); i++) {
        std::wstring currentChar = words[i];
        pairs.push_back(std::make_pair(prevChar, currentChar));
        prevChar = currentChar;
    }

    return pairs;
}

std::wstring SimpleTokenizer::whitespace_clean(std::wstring text) {
    // 使用正则表达式替换多个连续空白字符为单个空格
    text = std::regex_replace(text, std::wregex(L"\\s+"), L" ");
    // 去除字符串两端的空白字符
    text = std::regex_replace(text, std::wregex(L"^\\s+|\\s+$"), L"");

    return text;
}