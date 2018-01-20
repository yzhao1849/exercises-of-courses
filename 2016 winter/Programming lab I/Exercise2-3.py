# Formula of BMI=mass/height^2 (mass:kg, height:m)
weight=float(input("Your weight(kg): "))
height=float(input("Your height(m): "))
BMI=weight/(height**2)
print("You BMI is: {:.1f}".format(BMI))
if BMI<=15:
    print("You are very severely underweight!")
elif BMI<=16:
    print("You are severely underweight!")
elif BMI<=18.5:
    print("You are underweight!")
elif BMI<=25:
    print("Your weight is normal.")
elif BMI<=35:
    print("You are moderately obese!")
elif BMI<=40:
    print("You are severely obese!")
else:
    print("You are very severely obese!")

"""
Your weight(kg): 75
Your height(m): 1.74
You BMI is: 24.8
Your weight is normal.

Your weight(kg): 49
Your height(m): 1.85
You BMI is: 14.3
You are very severely underweight!

Your weight(kg): 128
Your height(m): 1.71
You BMI is: 43.8
You are very severely obese!
"""
