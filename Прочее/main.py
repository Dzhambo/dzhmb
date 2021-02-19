import os
import csv

class CarBase:
    def __init__(self, brand, photo_file_name, carrying):
        self.brand = brand
        self.photo_file_name = photo_file_name
        self.carrying = float(carrying)
        if not all(attr!='' for attr in (self.brand, self.photo_file_name, self.carrying)):
            raise ValueError
        self.ext = self.get_photo_file_ext()


    def get_photo_file_ext(self):
        extension = os.path.splitext(self.photo_file_name)[1]
        if extension not in ['.jpg', '.jpeg', '.png', '.gif']:
            raise ValueError
        return extension



class Car(CarBase):
    def __init__(self, brand, photo_file_name, carrying, passenger_seats_count):
        super().__init__(brand, photo_file_name, carrying)
        self.car_type = 'car'
        self.passenger_seats_count = int(passenger_seats_count)



class Truck(CarBase):
    def __init__(self, brand, photo_file_name, carrying, body_whl):
        super().__init__(brand, photo_file_name, carrying)
        try:
            body_parametres = list(map(float,body_whl.split('x')))
        except:
            body_parametres = [.0,.0,.0]
        if len(body_parametres)!=3:
            body_parametres = [.0, .0, .0]
        self.body_length = body_parametres[0]
        self.body_width = body_parametres[1]
        self.body_height = body_parametres[2]
        self.car_type = 'truck'

    def get_body_volume(self):
        body_volume = self.body_length * self.body_width * self.body_height
        return body_volume


class SpecMachine(CarBase):
    def __init__(self, brand, photo_file_name, carrying, extra):
        super().__init__(brand, photo_file_name, carrying)
        if extra == '':
            raise ValueError
        self.extra = extra
        self.car_type = 'spec_machine'


def get_car_list(csv_filename):
    car_list = []
    with open(csv_filename) as csv_fd:
        reader = csv.reader(csv_fd, delimiter=';')
        next(reader)
        car_types = {
            'car': lambda x: Car(x[1], x[3], x[5], x[2]),
            'truck': lambda x: Truck(x[1], x[3], x[5], x[4]),
            'spec_machine': lambda x: SpecMachine(x[1], x[3], x[5], x[6])}
        for row in reader:
            try:
                car_type = row[0]
                if car_type in car_types:
                    car_list.append(car_types[car_type](row))
            except (ValueError, IndexError):
                pass
    return car_list
