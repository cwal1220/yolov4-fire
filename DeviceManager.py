import RPi.GPIO as GPIO
import time

# Define GPIOs
L_IN1 = 23
L_IN2 = 24

# Init GPIOs
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(L_IN1, GPIO.OUT)
GPIO.setup(L_IN2, GPIO.OUT)

def fanStop():
    GPIO.output(L_IN1, GPIO.LOW)
    GPIO.output(L_IN2, GPIO.LOW)

def fanStart():
    GPIO.output(L_IN1, GPIO.LOW)
    GPIO.output(L_IN2, GPIO.HIGH)

if __name__ == '__main__':
    fanStart()
    time.sleep(3)
    fanStop()

