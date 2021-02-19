class Value:
    def __init__(self):
        self.value = None

    @staticmethod
    def amount_after_commission(value, commission):
        return value*(1 - commission)

    def __get__(self,obj,obj_type):
        return self.value

    def __set__(self,obj,value):
        self.value = self.amount_after_commission(value, obj.commission)

