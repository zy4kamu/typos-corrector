import cpp_bindings

if __name__ == '__main__':
    cpp_bindings.generate_cpp_bindings()

    print cpp_bindings.decompress('osterdokstrat')
    print cpp_bindings.decompress('amsterdamveg')
    print cpp_bindings.decompress('surinameplein')
    print cpp_bindings.decompress('dummy street')
