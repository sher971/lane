#MOTOR CLASS
import RPi.GPIO as GPIO
from time import sleep
in1 = 4
in2 = 17
in3 = 18
in4 = 27
en = 22
en2 = 23
temp1 = 1
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en, GPIO.OUT)
GPIO.setup(en2, GPIO.OUT)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)
p1 = GPIO.PWM(en, 1000)
p1.stop()
p2 = GPIO.PWM(en2, 1000)
p2.stop()
print("r-run s-stop f-forward b-backward l-low m-medium h-high dm-frontmiddle dr-frontright dl-frontleft e-exit")
def frontmiddle():
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)
    p2.stop()
def frontright():
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)
    p2.start(100)
def frontleft():
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    p2.start(100)
def forward(speed=50,time=0):
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    p1.start(speed)
    sleep(time)
def backward(speed=50,time=0):
    p1.start(speed)
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.HIGH)
    frontmiddle()
    sleep(time)
def stop():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)
    p1.stop()
    p2.stop()
def fright(speed=50,time=0):
    forward(speed)
    frontright()
    sleep(time)
def fleft(speed=50,time=0):
    forward(speed)
    frontleft()
    sleep(time)
def bright(speed=50,time=0):
    backward(speed)
    frontright()
    sleep(time)
def bleft(speed=50,time=0):
    backward(speed)
    frontleft()
    sleep(time)
if __name__ == '__main__':    
    stop()