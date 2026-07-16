## Trying to Build one Couries Tracing Class using all oops Concept 
# with the help of Gemini
### Using Abstac Method

from abc import ABC,abstractmethod
import exrex

## Create the Abstract Class
class Courier(ABC):
    def __init__(self,weight):
        self.weight = weight
        ## Encapsulation, Producting the Tracking status Attribute
        self.__Tracking_Status = 'Processing'
        self.__tracking_number = exrex.getone(r'[A-Z]{2}\d{9}TN')

    #Doubt : As the absctrt can't initate why init function return
       # Ans : This help to inheri the common Attribute to other 
       # Subclass instead of wringing code in every call
    @abstractmethod
    def calculate_price(self):
        pass

    def update_tracking_Status(self,new_status):
        self.__Tracking_Status = new_status

    def get_tracking_status(self):
        return f"The current Tracking Status is {self.__Tracking_Status}"
    
    def get_tracking_number(self):
        return self.__tracking_number
    
## Inheritance : Inheri the abstract Class
class Normal_Parcel(Courier):
    ## New feature super() understanding
        ## Super() will help to inheritae the Parent
        #  if we have separate __ini__ funation in chaile class
    def __init__(self,weight):
        super().__init__(weight) ## This will copy the Init attribute
    ## Polymorphism , calculate the Normal Couries charge 10 rs per Kg
    def calculate_price(self):
        return f"The Cost is Rs : {10*self.weight}"
    
class Express_Parcel(Courier):
    ## Polymorphism , calculate the Normal Couries charge 20 rs per Kg
    def calculate_price(self):
        return f"The Cost is Rs : {20*self.weight}"


parcel1 = Normal_Parcel(5)
print(parcel1.calculate_price())
print(parcel1.get_tracking_number())
print(parcel1.get_tracking_status())
parcel1.update_tracking_Status("transist")
print(parcel1.get_tracking_status())
parcel1.update_tracking_Status("delivered")
print(f"Your parcel with tracking ref , {parcel1.get_tracking_number()} is successfully {parcel1.get_tracking_status()}")

print(parcel1.__tracking_number)

parcel2 = Express_Parcel(5)
print(parcel2.calculate_price())
print(parcel2.get_tracking_number())
print(parcel2.get_tracking_status())
parcel2.update_tracking_Status("transist")
print(parcel2.get_tracking_status())
parcel2.update_tracking_Status("delivered")
print(f"Your parcel with tracking ref , {parcel2.get_tracking_number()} is successfully {parcel2.get_tracking_status()}")
