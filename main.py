from machine import ADC, Pin
import time
import os
import dht

sound_sensor = ADC(Pin(26)) # Analog sound
vibration_sensor = ADC(Pin(27)) # Analog vibration
light_sensor = ADC(Pin(28)) # Analog light
pir_sensor = Pin(14, Pin.IN) # PIR motion
dht_sensor = dht.DHT11(Pin(15)) # DHT11 Temp humidity

log_file = "sleep_data.csv"

def sample_all_over_30s():
    sound_max = 0
    vib_max = 0
    motion_flag = 0
    samples = 300
    delay_ms = 100

    for _ in range(samples):
        s = sound_sensor.read_u16()
        v = vibration_sensor.read_u16()
        p = pir_sensor.value()

        if s > sound_max:
            sound_max = s
        if v > vib_max:
            vib_max = v
        if p == 1:
            motion_flag = 1

        time.sleep_ms(delay_ms)

    sound_v = round(sound_max / 65535 * 3.3, 4)
    vib_v = round(vib_max / 65535 * 3.3, 4)

    return sound_v, vib_v, motion_flag


def read_dht():
    try:
        dht_sensor.measure()
        t = dht_sensor.temperature()
        h = dht_sensor.humidity()
        if -20 <= t <= 60 and 0 <= h <= 100:
            return t, h
    except:
        pass
    return "", ""

def read_light():
    val = light_sensor.read_u16()
    return round(val / 65535 * 3.3, 4)

def timestamp(start_time):
    elapsed = time.ticks_diff(time.ticks_ms(), start_time) // 1000
    h = elapsed // 3600
    m = (elapsed % 3600) // 60
    s = elapsed % 60
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)

if log_file not in os.listdir():
    with open(log_file, "w") as f:
        f.write("time,sound_max,vibration_max,light,temp,humidity,pir_motion\n")

print("Logging sensor data...")

start_time = time.ticks_ms()
while True:
    if time.ticks_diff(time.ticks_ms(), start_time) > 9 * 3600 * 1000:
        print("Logging complete")
        break

    ts = timestamp(start_time)
    sound_val, vib_val, motion_val = sample_all_over_30s()
    light_val = read_light()
    temp_val, humid_val = read_dht()

    row = f"{ts},{sound_val},{vib_val},{light_val},{temp_val},{humid_val},{motion_val}\n"
    
    with open(log_file, "a") as f:
        f.write(row)

    print(f"\nData Logged at {ts}")
    print(f"Sound Max Voltage: {sound_val} V")
    print(f"Vibration Max Voltage: {vib_val} V")
    print(f"Light Level: {light_val} V")
    print(f"PIR Motion Detected: {motion_val}")
    if temp_val != "":
        print(f"Temperature: {temp_val}")
        print(f"Humidity: {humid_val}")
    else:
        print("Temperature / Humidity: DHT Sensor Read Error")
