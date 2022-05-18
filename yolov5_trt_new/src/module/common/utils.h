#ifndef UTILS_H
#define UTILS_H

#include <memory>

namespace TRT{

template <typename T>
struct TrtDestroyer
{
    void operator()(T* t)
    {
        t->destroy();
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroyer<T>>;

// template <typename T>
// using TrtSharedPtr = std::shared_ptr<T, TrtDestroyer<T>>;

static bool save_file(const string& file, const void* data, size_t length){

    FILE* f = fopen(file.c_str(), "wb");
    if (!f) return false;

    if (data && length > 0){
        if (fwrite(data, 1, length, f) != length){
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

}

#endif
