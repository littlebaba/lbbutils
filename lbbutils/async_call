"""异步调用

    开启一个子线程去执行该函数

    :parameter
        fnc (function): 执行方法
        callback (function): 回调方法
    :return
        pass
    Example:
        def multi(x):
            print(5 * x)

        @async_call(callback=multi)
        def add(a, b):
            time.sleep(5)
            return a + b

        def subtract(a, b):
            print(a - b)

        x=add(1,2)
        y=subtract(5, 3)

        Run:
            2
            15
"""
import threading
import time


class AsyncCall(object):
    def __init__(self, callable, callback=None):
        self.callable = callable
        self.callback = callback
        self.result = None

    def __call__(self, *args, **kwargs):
        thread = threading.Thread(target=self.run, args=args, kwargs=kwargs)
        thread.start()

    def run(self, *args, **kwargs):
        self.result = self.callable(*args, **kwargs)
        if self.callback:
            self.result = self.callback(self.result)


def async_call(callback):
    def deco(fnc):
        def __wrapper(*args, **kwargs):
            callback_ = AsyncCall(fnc, callback)(*args, **kwargs)
            return callback_

        return __wrapper

    return deco


if __name__ == '__main__':
    def multi(x):
        print(5 * x)


    @async_call(callback=multi)
    def add(a, b):
        time.sleep(5)
        return a + b


    def subtract(a, b):
        print(a - b)


    x = add(1, 2)
    y = subtract(5, 3)
