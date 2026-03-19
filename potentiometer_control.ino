#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
void setup()
{
Serial.begin(9600);
pwm.begin();
pwm.setPWMFreq(60);
}
void loop() {
// Servo 0
int potValue0 = analogRead(A0);
int servoPos0 = map(potValue0, 0, 1023, 125, 575);
pwm.setPWM(0, 0, servoPos0);
// Servo 1
int potValue1 = analogRead(A1);
int servoPos1 = map(potValue1, 0, 1023, 125, 575);
pwm.setPWM(1, 0, servoPos1);
// Servo 2
int potValue2 = analogRead(A2);
int servoPos2 = map(potValue2, 0, 1023, 125, 575);
pwm.setPWM(2, 0, servoPos2);
// Servo 3
int potValue3 = analogRead(A3);
int servoPos3 = map(potValue3, 0, 1023, 125, 575);
pwm.setPWM(3, 0, servoPos3);
// Servo 4
int potValue4 = analogRead(A4);
int servoPos4 = map(potValue4, 0, 1023, 125, 575);
pwm.setPWM(4, 0, servoPos4);
// Servo 5
int potValue5 = analogRead(A5);
int servoPos5 = map(potValue5, 0, 1023, 125, 575);
pwm.setPWM(5, 0, servoPos5);
47
delay(20);
}