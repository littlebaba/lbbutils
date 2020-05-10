class EasyDict(dict):

    def __init__(self, d=None, **kwargs):
        if d == None:
            d = {}
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)



if __name__ == '__main__':
    d = EasyDict({'name': '李恒','age':19})
    d.phone = '100'
    dp=d.phone
    a = 1
