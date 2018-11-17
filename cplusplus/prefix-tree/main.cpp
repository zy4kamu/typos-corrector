#include "prefix-tree-builder.h"
#include "prefix-tree.h"

#include <iostream>
#include <fstream>

int main() {
    PrefixTreeBuilder builder;
    builder.add("hello");
    builder.add("help");
    builder.add("forest");
    builder.add("flower");
    std::vector<char> content = builder.to_string();
    {
      std::ofstream writer("/tmp/prefix-tree-test-file", std::ios::binary);
      writer.write(content.data(), content.size());
    }
    PrefixTree prefix_tree("/tmp/prefix-tree-test-file");

    prefix_tree.reset_pass();
    std::cout << prefix_tree.check("hel") << std::endl;
    prefix_tree.reset_pass();
    std::cout << prefix_tree.check("help") << std::endl;
    prefix_tree.reset_pass();
    std::cout << prefix_tree.check("help") << std::endl;
    prefix_tree.reset_pass();
    std::cout << prefix_tree.check("f") << std::endl;
    prefix_tree.reset_pass();
    std::cout << prefix_tree.check("flowe") << std::endl;
    prefix_tree.reset_pass();
    std::cout << prefix_tree.check("flowet") << std::endl;
}
