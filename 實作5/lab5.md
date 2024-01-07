# 嵌入式系統 - 實作5: Servo伺服馬達, 溫度感應器 + LCD 顯示器


## Lab 5-1 請使用兩個伺服馬達同步從 0 度逐步掃描到 180 度之後再逐步掃描回0度, 每步的間隔時間為50ms (0.05秒)

### Circuit

![image](https://github.com/Grace-TA/ES-Fall2021/blob/main/image/servo.gif)

### Code

![image](https://user-images.githubusercontent.com/89304181/138576789-3d0858a9-5b8e-4412-a713-961d61791e5b.png)

````C
// Developed for Embedded Sytem, VNU, Fall 2021

#include <Servo.h>

int pos = 0;

Servo servo_9;
Servo servo_8;
void setup()
{
  servo_9.attach(9, 500, 2500);
  servo_8.attach(8, 500, 2500);  
}

void loop()
{
// 從 0 到 180 度逐步掃描伺服, 1度/步
  for (pos = 0; pos <= 180; pos += 1) {

    servo_9.write(pos);
    servo_8.write(pos);       

    delay(50); // 等50ms (0.05秒)
  }
  
  for (pos = 180; pos >= 0; pos -= 1) {
// 從 180 到 0 度逐步掃描伺服, 1度/步
    servo_9.write(pos);
    servo_8.write(pos);        

    delay(50); // 等50ms (0.05秒)
  }
}

````


## Lab 5-2 LCD顯示溫度感應器的溫度;若溫度<38 綠LED亮; 若大於38度, 紅色LED亮

### Circuit

![image](https://github.com/Grace-TA/ES-Fall2021/blob/main/image/TempSensor.png)


### Code
````C
// For Embedded System course, VNU, Fall 2021 

#include <LiquidCrystal.h>

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

void setup() {
  lcd.begin(16, 2);
  Serial.begin(9600);	
  pinMode(A1, INPUT); // Read analog voltage (2^10)

}

void loop() {
  int reading = analogRead(A1);  // read analog level (2^10)
  lcd.setCursor(0,0);  
  lcd.print("TMP Sensor Demo");

  float voltage = (reading/1024.0) * 5.0; // Convert to voltage
  
  // converting from 10mv per degree with 500mV offset  
  float tempC = (voltage - 0.5) * 100.0; 
  lcd.setCursor(0,1);
  lcd.print("Tmp:");
  lcd.print(tempC);
  lcd.print(" C");
  delay(500);
  lcd.clear();
  Serial.println(reading);
  Serial.println(voltage);  
}
````

### Plus LED Control

![image](https://user-images.githubusercontent.com/89304181/138580142-b286feb8-36db-49ae-9684-a780e3d3505d.png)
