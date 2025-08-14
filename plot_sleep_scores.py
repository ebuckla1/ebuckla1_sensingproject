import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

input_file = "sleep_data.csv"
model_file = "sleep_score_windowed.pkl"
scaler_file = "feature_scaler.pkl"
window_minutes = 15

with open(model_file, "rb") as f:
    model = pickle.load(f)
with open(scaler_file, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()
df['time'] = pd.to_timedelta(df['time'])
df = df.set_index('time')

def extract_features(chunk):
    return {
        'motion_pct': chunk['pir_motion'].mean() * 100,
        'sound_avg': chunk['sound_max'].mean(),
        'sound_max': chunk['sound_max'].max(),
        'sound_std': chunk['sound_max'].std(),
        'vib_avg': chunk['vibration_max'].mean(),
        'vib_max': chunk['vibration_max'].max(),
        'light_avg': chunk['light'].mean(),
        'dark_pct': (chunk['light'] < 0.05).mean() * 100,
        'temp_avg': chunk['temp'].mean(),
        'temp_std': chunk['temp'].std(),
        'humidity_avg': chunk['humidity'].mean(),
        'humidity_std': chunk['humidity'].std(),
    }

grouped = df.groupby(df.index.floor(f"{window_minutes}min"))
window_scores = []
window_times = []

for window_start, chunk in grouped:
    if len(chunk) > 0:
        feats = extract_features(chunk)
        feats_df = pd.DataFrame([feats])
        feats_scaled = scaler.transform(feats_df)
        feats_scaled_df = pd.DataFrame(feats_scaled, columns=feats_df.columns)
        score = model.predict(feats_scaled_df)[0]
        window_scores.append(score)
        window_times.append(window_start)

overall_score = np.mean(window_scores)
print(f"Overall Mean Sleep Score: {overall_score:.1f} / 100")

t_hours_num = [td.total_seconds() / 3600 for td in df.index]
min_x = min(t_hours_num)
max_x = max(t_hours_num)


hour_ticks = np.arange(0, int(np.floor(max_x)) + 1, 1)

def hour_fmt(x, pos):
    h = int(x)
    m = int((x - h) * 60)
    return f"{h:02d}:{m:02d}"


fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
axs[0].set_title("Sleep Data with Predicted Sleep Scores", pad=40)

axs[0].semilogy(t_hours_num, df['sound_max'], '-')
axs[0].set_ylabel("Sound (V, log)")

axs[1].semilogy(t_hours_num, df['vibration_max'], '-')
axs[1].set_ylabel("Vibration (V, log)")

light_plot = df['light'].copy()
light_plot[light_plot <= 0] = np.nan
axs[2].semilogy(t_hours_num, light_plot, '-')
axs[2].set_ylabel("Light (V, log)")

axs[3].plot(t_hours_num, df['temp'], '-')
axs[3].set_ylabel("Temp (°C)")

axs[4].plot(t_hours_num, df['humidity'], '-')
axs[4].set_ylabel("Humidity (%)")

axs[5].step(t_hours_num, df['pir_motion'], '-')
axs[5].set_ylabel("Motion")
axs[5].set_xlabel("Time (HH:MM)")
axs[5].set_ylim([-0.1, 1.1])


for i, start_time in enumerate(window_times):
    start_hour = start_time.total_seconds() / 3600
    mid_hour = start_hour + (window_minutes / 60) / 2

    if i > 0:
        for ax in axs:
            ax.axvline(start_hour, color='gray', linestyle='--', linewidth=0.5)

    axs[0].text(mid_hour,
                max(df['sound_max']) * 1.8,
                f"{int(round(window_scores[i]))}",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='red')


pad = 0.02 * (max_x - min_x)
for ax in axs:
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_xticks(hour_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(hour_fmt))


axs[0].text(max_x * 1.02,
            max(df['sound_max']) * 4,
            f"Overall: {int(round(overall_score))}",
            ha='right', va='bottom',
            fontsize=14, fontweight='bold',
            color='blue')

plt.subplots_adjust(top=0.90, hspace=0.25)
plt.show()
