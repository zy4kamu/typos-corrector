#include "prefix-tree-builder.h"

PrefixTreeBuilderNode& PrefixTreeBuilderNode::operator[](char letter) {
    return content[letter];
}

std::vector<char> PrefixTreeBuilderNode::to_string() const {
   // structure of the tree is:
   // 1 byte: number of transitions (
   // n bytes: letter transitions
   // 4 * bytes: shifts
   // subtrees
   size_t total_size = 1;
   std::vector<char> letters;
   std::vector<std::vector<char>> subtrees;
   for (const auto& item : content) {
       letters.push_back(item.first);
       subtrees.push_back(item.second.to_string());
       total_size += 5 + subtrees.back().size();
   }

   std::vector<char> result(total_size);
   char* data = result.data();

   // copy number of subtrees
   uint8_t int_count = static_cast<uint8_t>(letters.size());
   std::memcpy(data, &int_count, 1);

   // copy transitions
   std::memcpy(data + 1, letters.data(), letters.size());

   uint32_t counter = 1 + 5 * letters.size();
   for (size_t i = 0; i < letters.size(); ++i) {
       // copy shift
       std::memcpy(data + 1 + letters.size() + 4 * i, &counter, 4);
       // copy sbutress
       std::memcpy(data + counter, subtrees[i].data(), subtrees[i].size());
       counter += subtrees[i].size();
   }
   return result;
}

void PrefixTreeBuilder::add(const std::string& message) {
  PrefixTreeBuilderNode* node = &root;
  for (const char letter : message) {
      node = &(*node)[letter];
  }
}

std::vector<char> PrefixTreeBuilder::to_string() const {
  return root.to_string();
}
