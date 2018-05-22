#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "prefix-tree.h"

namespace po = boost::program_options;

namespace {

const int32_t A_INT = static_cast<int32_t>('a');
const int32_t Z_INT = static_cast<int32_t>('z');
const size_t MIN_TOKEN_FREQUENCY = 5;

bool acceptable(char ch) {
    return ch == ' ' || ('a' <= ch && ch <= 'z');
}

int32_t to_int(char ch) {
    assert(acceptable(ch));
    if (ch == ' ') {
       return Z_INT - A_INT + 1;
    }
    return static_cast<int32_t>(ch) - A_INT;
}

char to_char(int32_t number) {
    assert(0 <= number && number <= Z_INT - A_INT + 1);
    return number == Z_INT - A_INT + 1 ? ' ' : static_cast<char>(A_INT + number);
}

} // anonymous namespace 

class DataSetPreparer {
public:
    DataSetPreparer(const std::string& input_file, const std::string& output_prefix_tree_file, 
                    size_t min_chars, size_t max_chars)
        : output_prefix_tree_file(output_prefix_tree_file), min_chars(min_chars), max_chars(max_chars) {
        if (!boost::filesystem::exists(input_file)) {
            throw std::runtime_error("absent input file " + input_file);
        }
        reader.open(input_file);
    }

    DataSetPreparer(const std::string& input_file, const std::string& output_folder, 
                    double mistake_probability, size_t min_chars, size_t max_chars, size_t batch_size)
        : output_folder(output_folder), mistake_probability(mistake_probability), min_chars(min_chars)
        , max_chars(max_chars), batch_size(batch_size), generator(1) {
        if (!boost::filesystem::exists(input_file)) {
            throw std::runtime_error("absent input file " + input_file);
        }
        reader.open(input_file);
        if (boost::filesystem::exists(output_folder)) {
            if (!boost::filesystem::remove_all(output_folder)) {
                throw std::runtime_error("coudln't delete output folder " + output_folder);
            }
        }
        if (!boost::filesystem::create_directories(output_folder)) {
            throw std::runtime_error("couldn't create output folder " + output_folder);
        }
        clean_writer.open((this->output_folder / "clean-0").string(), std::ios::binary);
        contaminated_writer.open((this->output_folder / "contaminated-0").string(), std::ios::binary);
    }

    void prepare() {
        size_t batch_index = 0;
        size_t counter = 0;
        std::string token;
        while (reader >> token && token.size() > 0) {
            token = clean_token(token);
            if (token.size() < min_chars || token.size() > max_chars) {
                continue;
            }
            std::string contaminated = contaminate(token);
            if (contaminated.size() < min_chars || contaminated.size() > max_chars) {
                continue;
            }
            print_token(token, clean_writer);
            print_token(contaminated, contaminated_writer);

            if (++counter == batch_size) {
                counter = 0;
                clean_writer.close();
                clean_writer.open((output_folder / ("clean-" + std::to_string(++batch_index))).string());
                contaminated_writer.close();
                contaminated_writer.open((output_folder / ("contaminated-" + std::to_string(batch_index))).string());
            }
        }

        if (counter != 0) {
            clean_writer.close();
            contaminated_writer.close();
            boost::filesystem::remove((output_folder / ("clean-" + std::to_string(batch_index))).string());
            boost::filesystem::remove((output_folder / ("contaminated-" + std::to_string(batch_index))).string());
        }
    }

    void build_prefix_tree() {
        PrefixTree prefix_tree;
        std::unordered_map<std::string, size_t> dictionary;
        std::string token;
        while (reader >> token && token.size() > 0) {
            token = clean_token(token);
            if (token.size() < min_chars || token.size() > max_chars) {
                continue;
            }
            ++dictionary[token];
        }
        for (const auto& item : dictionary) {
            if (item.second > MIN_TOKEN_FREQUENCY) {
                prefix_tree.add(item.first);
            }
        }
        std::cout << "saving tree to " << output_prefix_tree_file << std::endl;
        prefix_tree.save(output_prefix_tree_file);
    }

private:
    std::string clean_token(const std::string& token) {
        std::string cleaned;
        for (char ch: token) {
            ch = std::tolower(ch);
            if (acceptable(ch)) {
                cleaned += ch;
            }
        }
        return cleaned;
    }

    void print_token(const std::string& token, std::ofstream& writer) {
        std::vector<int32_t> converted(max_chars, Z_INT - A_INT + 1);
        std::transform(token.begin(), token.end(), converted.begin(), to_int);
        writer.write((char*)converted.data(), max_chars * sizeof(int32_t));
        writer.flush();
    }

    std::string swap_random_chars(const std::string& token) {
        if (token.size() < 2) { 
            return token; 
        }
        std::uniform_int_distribution<size_t> distribution(0, token.size() - 2);
        size_t index = distribution(generator);
        return token.substr(0, index) + token[index + 1] + token[index] + token.substr(index + 2);
    }

    std::string replace_random_char(const std::string& token) {
        std::uniform_int_distribution<size_t> char_distribution(A_INT, Z_INT);
        char letter = char_distribution(generator);

        std::uniform_int_distribution<size_t> distribution(0, token.size() - 1);
        size_t index = distribution(generator);
        if (index == token.size()) {
            return token + letter;
        }
        return token.substr(0, index) + letter + token.substr(index + 1);
    }

    std::string add_random_char(const std::string& token) {
        std::uniform_int_distribution<size_t> char_distribution(A_INT, Z_INT);
        char letter = char_distribution(generator);

        std::uniform_int_distribution<size_t> distribution(0, token.size());
        size_t index = distribution(generator);
        return token.substr(0, index) + letter + token.substr(index);
    }

    std::string remove_random_char(const std::string& token) {
        std::uniform_int_distribution<size_t> distribution(0, token.size() - 1);
        size_t index = distribution(generator);
        return token.substr(0, index) + token.substr(index + 1);
    }

    std::string contaminate(const std::string& token) {
        std::bernoulli_distribution contaminate_distribution(mistake_probability);
        std::uniform_int_distribution<int> distribution(0, 3);
        std::string contaminated_token = token;
        for (size_t i = 0; i < token.size(); ++i) {
            if (contaminate_distribution(generator)) {
                int type = distribution(generator);
                switch (type) {
                    case 0: 
                        contaminated_token = swap_random_chars(contaminated_token);
                        break;
                    case 1: 
                        contaminated_token = replace_random_char(contaminated_token);
                        break;
                    case 2: 
                        contaminated_token = add_random_char(contaminated_token);
                        break;
                    case 3: 
                        contaminated_token = remove_random_char(contaminated_token);
                        break;
                }
            }
        }
        return contaminated_token;
    }

    boost::filesystem::path output_folder;
    std::string output_prefix_tree_file;

    double mistake_probability;
    size_t min_chars;
    size_t max_chars;
    size_t batch_size;

    std::ifstream reader;
    std::ofstream clean_writer;
    std::ofstream contaminated_writer;

    std::mt19937 generator;
};

void test_dataset_preparer(const std::string& output_folder, size_t max_chars, size_t batch_size) {
    std::ifstream clean_reader(output_folder + "/clean-7", std::ios::binary);
    std::ifstream contaminated_reader(output_folder + "/contaminated-7", std::ios::binary);
    std::vector<int32_t> encoded(max_chars);
    std::vector<char> decoded(max_chars);

    for (size_t i = 0; i < batch_size; ++i) { 
        clean_reader.read((char*)encoded.data(), max_chars * sizeof(int32_t));
        std::transform(encoded.begin(), encoded.end(), decoded.begin(), to_char);
        std::string clean_token(decoded.data(), max_chars);

        contaminated_reader.read((char*)encoded.data(), max_chars * sizeof(int32_t));
        std::transform(encoded.begin(), encoded.end(), decoded.begin(), to_char);
        std::string contaminated_token(decoded.data(), max_chars);

        std::cout << clean_token << " " << contaminated_token << std::endl;
    }
}

void test_prefix_tree(const std::string& input_file) {
    PrefixTree tree(input_file);
    std::vector<std::string> prefixes { "hello", "xoxoxox", "moscow", "mother", "motherr", 
                                        "motherrrrr" }; 
    for (const std::string& prefix : prefixes) {
        std::cout << prefix << " " << tree.match(prefix) << std::endl;
    }
}

int main(int argc, char* argv[]) {

    std::string command;
    std::string text;
    std::string output_folder;
    double mistake_probability;
    size_t min_chars;
    size_t max_chars;
    size_t batch_size;
    std::string prefix_tree_file;

    po::options_description desc("Prepares batches for typos correction train and test");
    try {
        desc.add_options()("command,c", po::value<std::string>(&command)->required(), 
                           "task to do: dataset/prefix-tree");
        desc.add_options()("text,t", po::value<std::string>(&text)->required(), 
                           "input text text file");
        desc.add_options()("output-folder,o", po::value<std::string>(&output_folder)->default_value("dataset"),
                           "output folder with binary batches");
        desc.add_options()("prefix-tree-file,p", po::value<std::string>(&prefix_tree_file)->default_value("prefix-tree"),
                           "output file with binary prefix tree");
        desc.add_options()("mistake-probability,p", po::value<double>(&mistake_probability)->default_value(0.2),
                           "probabilty to make a mistake in each token");
        desc.add_options()("min-chars,mn", po::value<size_t>(&min_chars)->default_value(3), 
                           "min number of characters in token in order to take it into account");
        desc.add_options()("max-chars,mx", po::value<size_t>(&max_chars)->default_value(10),
                           "max number of characters in token in order to take it into account");
        desc.add_options()("batch-size,b", po::value<size_t>(&batch_size)->default_value(500),
                           "number of tokens in batch");
        po::variables_map variables_map;
        po::store(po::parse_command_line(argc, argv, desc), variables_map);
        po::notify(variables_map);    
    } catch (std::exception&) {
        std::cout << desc << std::endl;
        exit(0);
    }

    if (command == "dataset") {
        {
            DataSetPreparer preparer(text, output_folder, mistake_probability, min_chars, max_chars,
                                     batch_size);
            preparer.prepare();
        }
        test_dataset_preparer(output_folder, max_chars, batch_size);
    } else if (command == "prefix-tree") {
        DataSetPreparer preparer(text, prefix_tree_file, min_chars, max_chars);
        preparer.build_prefix_tree();
        test_prefix_tree(prefix_tree_file);
    } else {
        std::cerr << "Invalid command parameter" << std::endl;
        std::cout << desc << std::endl;
    }
}
