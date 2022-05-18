#include "ilogger.h"
#include <string>
#include <stdarg.h>

using namespace std;

namespace iLogger{
static string file_name(const string& path, bool include_suffix){

    if (path.empty()) return "";

    int p = path.rfind('/');

    p += 1;

    //include suffix
    if (include_suffix)
        return path.substr(p);

    int u = path.rfind('.');
    if (u == -1)
        return path.substr(p);

    if (u <= p) u = path.size();
    return path.substr(p, u - p);
}

static const char* level_string(LogLevel level){
    switch (level){
        case LogLevel::Debug: return "debug";
        case LogLevel::Verbose: return "verbo";
        case LogLevel::Info: return "info";
        case LogLevel::Warning: return "warn";
        case LogLevel::Error: return "error";
        case LogLevel::Fatal: return "fatal";
        default: return "unknow";
    }
}

void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...){

    if(level > CURRENT_LOG_LEVEL)
        return;

    va_list vl;
    va_start(vl, fmt);
    
    char buffer[2048];
    string filename = file_name(file, true);
    int n = snprintf(buffer, sizeof(buffer), "[%s][%s:%d]:", level_string(level), filename.c_str(), line);
    vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);

    fprintf(stdout, "%s\n", buffer);
    if (level == LogLevel::Fatal) {
        fflush(stdout);
        abort();
    }
}

}