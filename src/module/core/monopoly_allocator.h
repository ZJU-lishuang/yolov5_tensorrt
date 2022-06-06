#ifndef MONOPOLY_ALLOCATOR_H
#define MONOPOLY_ALLOCATOR_H

#include <condition_variable>
#include <vector>
#include <mutex>
#include <memory>

template<class _ItemType>
class MonopolyAllocator{
public:
    class MonopolyData{
    public:
        std::shared_ptr<_ItemType>& data(){return data_;}
        void release(){manager_->release_one(this);}

    private:
        MonopolyData(MonopolyAllocator* pmanager){manager_ = pmanager;}

        friend class MonopolyAllocator;
        MonopolyAllocator* manager_ = nullptr;
        std::shared_ptr<_ItemType> data_;
        bool available_ = true;
    };
    typedef std::shared_ptr<MonopolyData> MonopolyDataPointer;

    MonopolyAllocator(int size){
        capacity_ = size;
        num_available_ = size;
        datas_.resize(size);

        for(int i = 0;i < size; ++i){
            datas_[i] = std::shared_ptr<MonopolyData>(new MonopolyData(this));
        }
    }

    virtual ~MonopolyAllocator(){
        run_ = false;
        cv_.notify_all(); // 唤醒所有的等待(wait)线程。如果当前没有等待线程，则该函数什么也不做。

        std::unique_lock<std::mutex> l(lock_);
        // lambda函数 [外部变量访问方式说明符](参数表){语句块}
        // 当收到其他线程通知且num_wait_thread_ == 0时，才解除阻塞
        // 当num_wait_thread_ ！= 0时，阻塞当前线程
        cv_exit_.wait(l,[&](){
            return num_wait_thread_ == 0;
        });
    }

    MonopolyDataPointer query(int timeout = 10000){
        std::unique_lock<std::mutex> l(lock_);
        if(!run_) return nullptr;

        if(num_available_ == 0){
            num_wait_thread_++;

            // 指定一个时间，超时后且返回值为true才解除阻塞
            auto state = cv_.wait_for(l,std::chrono::milliseconds(timeout),[&](){
                return num_available_ > 0 || !run_;
            });

            num_wait_thread_--;
            // 唤醒某个等待(wait)线程。如果当前没有等待线程，则该函数什么也不做，如果同时存在多个等待线程，则唤醒某个线程是不确定的(unspecified)。
            cv_exit_.notify_one();

            if(!state || num_available_ == 0 || !run_)
                return nullptr;
        }

        auto item = std::find_if(datas_.begin(),datas_.end(),[](MonopolyDataPointer& item){
            return item->available_;
        });
        if(item == datas_.end())
            return nullptr;

        (*item)->available_ = false;
        num_available_--;
        return *item;
    }

    int num_available(){
        return num_available_;
    }

    int capacity(){
        return capacity_;
    }

private:
    void release_one(MonopolyData* prq){
        std::unique_lock<std::mutex> l(lock_);
        if(!prq->available_){
            prq->available_ = true;
            num_available_++;
            cv_.notify_one();
        }

    }
    // 互斥锁
    std::mutex lock_;
    // 条件变量
    std::condition_variable cv_;
    std::condition_variable cv_exit_;
    std::vector<MonopolyDataPointer> datas_;
    int capacity_ = 0;
    /*
    volatile提醒编译器它后面所定义的变量随时都有可能改变，
    因此编译后的程序每次需要存储或读取这个变量的时候，告诉编译器对该变量不做优化，
    都会直接从变量内存地址中读取数据，从而可以提供对特殊地址的稳定访问。

    如果没有volatile关键字，则编译器可能优化读取和存储，
    可能暂时使用寄存器中的值，如果这个变量由别的程序更新了的话，
    将出现不一致的现象。
    （简洁的说就是：volatile关键词影响编译器编译的结果，
    用volatile声明的变量表示该变量随时可能发生变化，
    与该变量有关的运算，不要进行编译优化，以免出错）
    */
    volatile int num_available_ = 0;
    volatile int num_wait_thread_ = 0;
    volatile bool run_ = true;
};

#endif