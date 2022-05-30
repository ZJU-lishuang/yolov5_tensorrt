#ifndef ILOGGER_HPP
#define ILOGGER_HPP

namespace iLogger{

enum class LogLevel : int{
    Debug   = 5,
    Verbose = 4,
    Info    = 3,
    Warning = 2,
    Error   = 1,
    Fatal   = 0
};

/* 修改这个level来实现修改日志输出级别 */
#define CURRENT_LOG_LEVEL       LogLevel::Info
// 可变参数宏__VA_ARGS__: 宏可以接受可变数目的参数
#define INFOD(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Debug, __VA_ARGS__)
#define INFOV(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Verbose, __VA_ARGS__)
#define INFO(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Info, __VA_ARGS__)
#define INFOW(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Warning, __VA_ARGS__)
#define INFOE(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Error, __VA_ARGS__)
#define INFOF(...)			iLogger::__log_func(__FILE__, __LINE__, iLogger::LogLevel::Fatal, __VA_ARGS__)

void __log_func(const char* file, int line, LogLevel level, const char* fmt, ...);

}

#endif