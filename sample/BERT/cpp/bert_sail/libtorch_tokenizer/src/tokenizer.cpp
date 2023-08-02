//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include "unicode.h"
#include "uninorms.h"
#include <codecvt>
#include <algorithm>
#include <boost/regex/pending/unicode_iterator.hpp>
#include <boost/spirit/include/qi.hpp>
#include <cstdint>
#include "tokenizer.h"

using namespace std;
using namespace ufal::unilib;
using namespace boost;
using namespace spirit::qi;


map<std::string, unicode::category_t> categories = {
        {"Lu", unicode::Lu},
        {"Ll", unicode::Ll},
        {"Lt", unicode::Lt},
        {"Lm", unicode::Lm},
        {"Lo", unicode::Lo},
        {"Mn", unicode::Mn},
        {"Mc", unicode::Mc},
        {"Me", unicode::Me},
        {"Nd", unicode::Nd},
        {"Nl", unicode::Nl},
        {"No", unicode::No},
        {"Pc", unicode::Pc},
        {"Pd", unicode::Pd},
        {"Ps", unicode::Ps},
        {"Pe", unicode::Pe},
        {"Pi", unicode::Pi},
        {"Pf", unicode::Pf},
        {"Po", unicode::Po},
        {"Sm", unicode::Sm},
        {"Sc", unicode::Sc},
        {"Sk", unicode::Sk},
        {"So", unicode::So},
        {"Zs", unicode::Zs},
        {"Zl", unicode::Zl},
        {"Zp", unicode::Zp},
        {"Cc", unicode::Cc},
        {"Cf", unicode::Cf},
        {"Cs", unicode::Cs},
        {"Co", unicode::Co},
        {"Cn", unicode::Cn},
};

map<unicode::category_t, std::string> categories_rev;

std::string ltrim(std::string str)
{
    return regex_replace(str, regex("^\\s+"), std::string(""));
}

std::string rtrim(std::string str)
{
    return regex_replace(str, regex("\\s+$"), std::string(""));
}

std::string trim(std::string str)
{
    return ltrim(rtrim(str));
}

vector<std::string> split(const std::string &str, char delimiter)
{
    vector<std::string> internal;
    std::stringstream ss(str); // Turn the std::string into a stream.
    std::string tok;

    while (getline(ss, tok, delimiter))
    {
        internal.push_back(tok);
    }
    return internal;
}

map<std::string, int> read_vocab(const char *filename)
{
    map<std::string, int> vocab;
    int index = 0;
    unsigned int line_count = 1;
    ifstream fs8(filename);
    if (!fs8.is_open())
    {
        cout << "Could not open " << filename << endl;
        return vocab;
    }
    std::string line;
    // Read all the lines in the file
    while (getline(fs8, line))
    {
        // check for invalid utf-8 (for a simple yes/no check, there is also utf8::is_valid function)
        // std::string::iterator end_it = utf8::find_invalid(line.begin(), line.end());
        vocab.insert(pair<std::string, int>(std::string(line.begin(), line.end()), index));
        index++;
        line_count++;
    }
    return vocab;
}

vector<std::string> whitespace_tokenize(std::string text)
{
    vector<std::string> result;
    char delimeter = ' ';
    text = trim(text);
    if (text == "")
    {
        return result;
    }
    result = split(text, delimeter);
    return result;
}

bool _is_whitespace(char letter)
{
    if (letter == ' ' or letter == '\t' or letter == '\n' or letter == '\r')
        return true;
    long int cat = unicode::category(int(letter));
    if (cat == categories["Zs"])
        return true;
    return false;
}

bool _is_control(char letter)
{
    if (letter == '\t' or letter == '\n' or letter == '\r')
        return false;
    unicode::category_t cat = unicode::category(int(letter));
    std::string cat_ = categories_rev[cat];
    if (cat_[0] == 'C')
        return true;
    return false;
}

bool _is_punctuation(char letter)
{
    int cp = int(letter);
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126))
        return true;
    unicode::category_t cat = unicode::category(int(letter));
    std::string cat_ = categories_rev[cat];
    if (cat_[0] == 'P')
        return true;
    return false;
}

std::string BasicTokenizer::_clean_text(std::string text)
{
    std::string output;
    int len = 0;
    char *char_array = new char[text.length() + 1];
    strcpy(char_array, text.c_str());
    while (char_array[len] != '\0')
    {
        int cp = int(char_array[len]);
        if (cp == 0 or cp == 0xfffd or _is_control(char_array[len]))
            continue;
        if (_is_whitespace(char_array[len]))
            output = output + " ";
        else
            output = output + char_array[len];
        ++len;
    }
    return output;
}

vector<std::string> BasicTokenizer::_run_split_on_punc(std::string text)
{
    // vector<std::string> never_split = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"};
    if (find(never_split_.begin(), never_split_.end(), text) != never_split_.end())
    {
        vector<std::string> temp = {text};
        return temp;
    }
    int len_char_array = text.length();
    char *char_array = new char[text.length() + 1];
    strcpy(char_array, text.c_str());
    int i = 0;
    bool start_new_word = true;
    vector<vector<char>> output;
    while (i < len_char_array)
    {
        char letter = char_array[i];
        if (_is_punctuation(letter))
        {
            vector<char> temp = {letter};
            output.push_back(temp);
            start_new_word = true;
        }
        else
        {
            if (start_new_word)
            {
                vector<char> temp_2;
                output.push_back(temp_2);
            }
            start_new_word = false;
            output.back().push_back(letter);
        }
        i += 1;
    }
    vector<std::string> final_output;
    vector<vector<char>>::iterator ptr;
    for (ptr = output.begin(); ptr < output.end(); ptr++)
    {
        vector<char> out = *ptr;
        std::string word = "";
        vector<char>::iterator itr;
        for (itr = out.begin(); itr < out.end(); itr++)
        {
            word = word + *itr;
        }
        final_output.push_back(word);
    }
    return final_output;
}

std::string BasicTokenizer::_run_strip_accents(std::string text)
{
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    auto temp = conv.from_bytes(text);
    auto nfd = [](u32string str)
    {
        uninorms::nfd(str);
        return str;
    };
    auto text_ = nfd(temp);
    std::string output;
    int i = 0;
    int len_char_array = text_.length() + 1;
    char *char_array = new char[text_.length() + 1];
    int j;
    for (j = 0; j < len_char_array; j++)
    {
        char_array[j] = text_[j];
    }
    while (i < len_char_array)
    {
        long int cat = unicode::category(int(char_array[i]));
        if (cat == categories["Mn"])
        {
            i++;
            continue;
        }
        // if (_is_punctuation(char_array[i]))
        // {
        //     i++;
        //     continue;
        // }
        output = output + char_array[i];
        i++;
    }
    return output;
}

std::string BasicTokenizer::utf8chr(int cp)
{
    char c[5] = {0x00, 0x00, 0x00, 0x00, 0x00};
    if (cp <= 0x7F) { c[0] = cp; }
    else if (cp <= 0x7FF)
    {
        c[0] = (cp >> 6) + 192;
        c[1] = (cp & 63) + 128;
    }
    else if (0xd800 <= cp && cp <= 0xdfff) {} //invalid block of utf8
    else if (cp <= 0xFFFF)
    {
        c[0] = (cp >> 12) + 224;
        c[1] = ((cp >> 6) & 63) + 128;
        c[2] = (cp & 63) + 128;
    }
    else if (cp <= 0x10FFFF)
    {
        c[0] = (cp >> 18) + 240;
        c[1] = ((cp >> 12) & 63) + 128;
        c[2] = ((cp >> 6) & 63) + 128;
        c[3] = (cp & 63) + 128;
    }
    return std::string(c);
}

std::string BasicTokenizer::
_tokenize_chinese_chars(std::string text)
{
    auto &&utf8_text = text;
    u8_to_u32_iterator<std::string::iterator>
            tbegin(utf8_text.begin()), tend(utf8_text.end());
    vector<uint32_t> result;
    parse(tbegin, tend, *standard_wide::char_, result);
    std::string output;
    for (auto &&code_point : result)
    {
        int cp = code_point;
        if (_is_chinese_char(cp))
        {
            output += " ";
            output.append(utf8chr(code_point));
            output += " ";
        }
        else
        {
            output.append(utf8chr(code_point));
        }
//        ++len;
    }

    return output;
}

bool BasicTokenizer::_is_chinese_char(int cp)
{
    if (
            (cp >= 0x4E00 && cp <= 0x9FFF)
            || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x20000 && cp <= 0x2A6DF)
            || (cp >= 0x2A700 && cp <= 0x2B73F)
            || (cp >= 0x2B740 && cp <= 0x2B81F)
            || (cp >= 0x2B820 && cp <= 0x2CEAF)
            || (cp >= 0xF900 && cp <= 0xFAFF)
            || (cp >= 0x2F800 && cp <= 0x2FA1F) || cp == 0x3002 || cp == 0xFF1F || cp == 0xFF01 || cp == 0xFF0C ||
            cp == 0x3001 || cp == 0xFF1B || cp == 0xFF1A || cp == 0x300C || cp == 0x300D || cp == 0x300E ||
            cp == 0x300F || cp == 0x2018 || cp == 0x2019 || cp == 0x201C || cp == 0x201D || cp == 0xFF08 ||
            cp == 0xFF09 || cp == 0x3014 || cp == 0x3015 || cp == 0x3010 || cp == 0x3011 || cp == 0x2014 ||
            cp == 0x2026 || cp == 0x2013 || cp == 0xFF0E || cp == 0x300A || cp == 0x300B || cp == 0x3008 || cp == 0x3009
            )
        return true;
    else
        return false;
}


vector<std::string> BasicTokenizer::tokenize(std::string text)
{
//    text = _clean_text(text);
    text = _tokenize_chinese_chars(text);
    vector<std::string> orig_tokens = whitespace_tokenize(text);
    vector<std::string> split_tokens;
    vector<std::string>::iterator itr;
    for (itr = orig_tokens.begin(); itr < orig_tokens.end(); itr++)
    {
        std::string temp = *itr;
        if (do_lower_case_ and not bool(find(never_split_.begin(), never_split_.end(), *itr) != never_split_.end()))
        {
            transform(temp.begin(), temp.end(), temp.begin(), [](unsigned char c) { return std::tolower(c); });
            temp = _run_strip_accents(temp);
        }
        vector<std::string> split = _run_split_on_punc(temp);
        split_tokens.insert(split_tokens.end(), split.begin(), split.end());
    }
    std::string temp_text;
    vector<std::string>::iterator ptr;
    for (ptr = split_tokens.begin(); ptr < split_tokens.end(); ptr++)
    {
        temp_text = temp_text + " " + *ptr;
    }
    return whitespace_tokenize(temp_text);
}

void BasicTokenizer::truncate_sequences(
        vector<std::string> &tokens_A, vector<std::string> &tokens_B, const char *truncation_strategy = "longest_first",
        int max_seq_length = 509)
{
    int length = tokens_A.size() + tokens_B.size();
    if (strcmp(truncation_strategy, "longest_first") == 0)
    {
        while (length > max_seq_length)
        {
            if (tokens_A.empty() || tokens_A.size() > tokens_B.size())
            {
                tokens_A.pop_back();
            }
            else
            {
                tokens_B.pop_back();
            }
            --length;
        }
    }
    else if (strcmp(truncation_strategy, "only_first") == 0)
    {
        while (length > max_seq_length && !tokens_A.empty())
        {
            tokens_A.pop_back();
            --length;
        }
    }
    else if (strcmp(truncation_strategy, "only_second") == 0)
    {
        while (length > max_seq_length && !tokens_B.empty())
        {
            tokens_B.pop_back();
            --length;
        }
    }
    else if (strcmp(truncation_strategy, "do_not_truncate") == 0)
    {
        assert((length < max_seq_length));
    }
    else
    {
        cerr << "invalid truncation strategy.  skipping trancation" << endl;
    }
}

void WordpieceTokenizer::add_vocab(map<std::string, int> vocab)
{
    vocab_ = vocab;
    unk_token_ = "[UNK]";
    max_input_chars_per_word_ = 100;
}

vector<std::string> WordpieceTokenizer::tokenize(std::string text)
{
    vector<std::string> output_tokens;
    vector<std::string> whitespace_tokens = whitespace_tokenize(text);
    vector<std::string>::iterator ptr;
    for (ptr = whitespace_tokens.begin(); ptr < whitespace_tokens.end(); ptr++)
    {
        // cout<<*ptr<<"\n";
        std::string token = *ptr;
        int len_char_array = token.length();
//        cout << len_char_array <<endl;
        char *char_array = new char[token.length() + 1];
        strcpy(char_array, token.c_str());
        if (len_char_array > max_input_chars_per_word_)
        {
            output_tokens.push_back(unk_token_);
            continue;
        }
        // cout<<len_char_array<<'\n';
        bool is_bad = false;
        int start = 0;
        vector<std::string> sub_tokens;
        while (start < len_char_array)
        {
            int end = len_char_array;
            std::string cur_substr = "";
            while (start < end)
            {
                std::string substr;
                for (int c = start; c < end; c++)
                    substr = substr + char_array[c];
                if (start > 0)
                    substr = "##" + substr;
                if (vocab_.count(substr) == 1)
                {
                    cur_substr = substr;
                    break;
                }
                end = end - 1;
            }
            if (cur_substr == "")
            {
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            start = end;
        }
        if (is_bad)
            output_tokens.push_back(unk_token_);
        else
        {
            output_tokens.insert(output_tokens.end(), sub_tokens.begin(), sub_tokens.end());
        }
    }
    return output_tokens;
}


void BertTokenizer::add_vocab(const char *vocab_file)
{
    vocab = read_vocab(vocab_file);
    for (map<std::string, int>::iterator i = vocab.begin(); i != vocab.end(); ++i)
        ids_to_tokens[i->second] = i->first;
    do_basic_tokenize_ = true;
    do_lower_case_ = false;
    wordpiece_tokenizer.add_vocab(vocab);
    maxlen_ = 512;
}

vector<std::string> BertTokenizer::tokenize(std::string text)
{
    vector<std::string> split_tokens;
    if (do_basic_tokenize_)
    {
        vector<std::string> temp_tokens = basic_tokenizer.tokenize(text);
        vector<std::string>::iterator ptr;
        for (ptr = temp_tokens.begin(); ptr < temp_tokens.end(); ptr++)
        {
            vector<std::string> subtokens = wordpiece_tokenizer.tokenize(*ptr);
            split_tokens.insert(split_tokens.end(), subtokens.begin(), subtokens.end());
        }
    }
    else
    {
        split_tokens = wordpiece_tokenizer.tokenize(text);
    }
    return split_tokens;
}

vector<float> BertTokenizer::convert_tokens_to_ids(vector<std::string> tokens)
{
    vector<float> ids;
    vector<std::string>::iterator ptr;
    for (ptr = tokens.begin(); ptr < tokens.end(); ptr++)
    {
        ids.push_back(float(vocab[*ptr]));
    }
    if (ids.size() > maxlen_)
        cout << "Token indices sequence length is longer than the specified maximum";
    return ids;
}

void
BertTokenizer::encode(std::string textA, std::string textB, vector<float> &input_ids, vector<float> &input_mask,
                      vector<float> &segment_ids, int max_seq_length, const char *truncation_strategy)
{
    BasicTokenizer basictokenizer;
    vector<std::string> tokens_A;
    vector<std::string> words = basictokenizer.tokenize(textA);
    vector<std::string> token;
    vector<std::string>::iterator itr;
    for (itr = words.begin(); itr < words.end(); itr++)
    {
        token = this->tokenize(*itr);
        tokens_A.insert(tokens_A.end(), token.begin(), token.end());
    }
    if (textB == "")
    {
        if (tokens_A.size() > max_seq_length - 2)
        {
            tokens_A.assign(tokens_A.begin(), tokens_A.begin() + max_seq_length - 2);
        }
        // insert "[CLS}"
        tokens_A.insert(tokens_A.begin(), "[CLS]");
        // insert "[SEP]"
        tokens_A.push_back("[SEP]");
        for (int i = 0; i < tokens_A.size(); i++)
        {
            segment_ids.push_back(0.0);
            input_mask.push_back(1.0);
        }
        input_ids = this->convert_tokens_to_ids(tokens_A);
        while (input_ids.size() < max_seq_length)
        {
            input_ids.push_back(0.0);
            input_mask.push_back(0.0);
            segment_ids.push_back(0.0);
        }
    }
    else
    {
        vector<std::string> tokens_B;
        words = basictokenizer.tokenize(textB);
        for (itr = words.begin(); itr < words.end(); itr++)
        {
            token = this->tokenize(*itr);
            tokens_B.insert(tokens_B.end(), token.begin(), token.end());
        }
        basictokenizer.truncate_sequences(tokens_A, tokens_B, truncation_strategy, max_seq_length - 3);
        // insert "[CLS}"
        tokens_A.insert(tokens_A.begin(), "[CLS]");
        // insert "[SEP]"
        tokens_A.push_back("[SEP]");
        for (int i = 0; i < tokens_A.size(); i++)
        {
            segment_ids.push_back(0.0);
            input_mask.push_back(1.0);
        }
        // insert "[SEP]"
        tokens_B.push_back("[SEP]");
        for (int i = 0; i < tokens_B.size(); i++)
        {
            segment_ids.push_back(0.0);
            input_mask.push_back(1.0);
        }
        tokens_A.insert(tokens_A.end(), tokens_B.begin(), tokens_B.end());
        // Padding
        input_ids = this->convert_tokens_to_ids(tokens_A);
        while (input_ids.size() < max_seq_length)
        {
            input_ids.push_back(0.0);
            input_mask.push_back(0.0);
            segment_ids.push_back(0.0);
        }
    }
    for (auto &&token:tokens_A)
    {
        cout << token << " ";
    }
    cout << endl;
}

