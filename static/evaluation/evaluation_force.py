import os
import sys
sys.path.append(os.getcwd())
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz

from util_rosbags import bag_to_array


def metric_signal_energy(F):
    return np.sum(np.abs(F)**2)/F.size

def metric_signal_smoothness(F):
    return np.std(np.gradient(F))

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def process_trial_folders(benchmark_data_path, metrics_file):
            # Find all trial folders
        trial_folders = [
            os.path.join(benchmark_data_path, folder)
            for folder in os.listdir(benchmark_data_path)
            if "trial" in folder.lower() and os.path.isdir(os.path.join(benchmark_data_path, folder))
        ]

        for trial_folder in trial_folders:
            # Find all subtask folders in the trial folder
            subtask_folders = [
                os.path.join(trial_folder, subfolder)
                for subfolder in os.listdir(trial_folder)
                if os.path.isdir(os.path.join(trial_folder, subfolder))
            ]

            for subtask_folder in subtask_folders:
                subtask_name = os.path.basename(subtask_folder)

                # Find all bag files in the subtask folder
                bag_files = sorted(glob.glob(os.path.join(subtask_folder, "*.bag")))

                force_data = []
                time_data = []

                for bag_file in bag_files:
                    # Read force data from the bag file
                    forces, times = bag_to_array(topic_name=rosbag_force_sensor_topic, path_to_bag=bag_file)
                    print(bag_file)
                    
                    if time_data:
                        # Adjust time to be continuous across multiple bag files
                        times += time_data[-1][-1]

                    force_data.append(forces)
                    time_data.append(times)

                # Concatenate all force and time data
                force_data = np.concatenate(force_data)
                time_data = np.concatenate(time_data)

                # Apply low-pass filter to force data
                order = 6
                fs = 1000.0         # sample rate, Hz
                cutoff = 5.0        # desired cutoff frequency of the filter, Hz

                Fx = force_data[:,0]
                Fy = force_data[:,1]
                Fz = force_data[:,2]

                Fx = butter_lowpass_filter(Fx, cutoff, fs, order)
                Fy = butter_lowpass_filter(Fy, cutoff, fs, order)
                Fz = butter_lowpass_filter(Fz, cutoff, fs, order)

                # Compute force in plane perpendicular to Fz
                Fxy = np.stack([Fx, Fy], axis=1)
                Fxy = np.linalg.norm(Fxy, axis=1)

                # Calculate metrics
                E_z = metric_signal_energy(F=Fz)
                E_xy = metric_signal_energy(F=Fxy)
                S_z = metric_signal_smoothness(F=Fz)
                S_xy = metric_signal_smoothness(F=Fxy)

                # Save metrics in a dictionary
                if 'metrics' not in locals():
                    metrics = {}
                if subtask_name not in metrics:
                    metrics[subtask_name] = {"E_z": [], "E_xy": [], "S_z": [], "S_xy": []}

                metrics[subtask_name]["E_z"].append(E_z)
                metrics[subtask_name]["E_xy"].append(E_xy)
                metrics[subtask_name]["S_z"].append(S_z)
                metrics[subtask_name]["S_xy"].append(S_xy)

                # Plot force data over time
                plt.figure()
                plt.plot(time_data, force_data[:, 0], label="Force X", color="red")
                plt.plot(time_data, force_data[:, 1], label="Force Y", color="green")
                plt.plot(time_data, force_data[:, 2], label="Force Z", color="blue")
                plt.title(f"Force Data for Subtask: {subtask_name}")
                plt.xlabel("Time [s]")
                plt.ylabel("Force [N]")
                plt.legend()
                plt.grid()

                # Save the plot
                plot_path = os.path.join(subtask_folder, f"{subtask_name}_force_plot.png")
                plt.savefig(plot_path)
                plt.close()

        # Save metrics to a numpy file for later use
        np.save(metrics_file, metrics)

def plot_qbit_metrics(benchmark_data_path, metrics_file):
    # After processing all subtasks, create boxplots for each metric
    metrics = np.load(metrics_file, allow_pickle=True).item()
    cm = 1 / 2.54  # centimeters to inches
    
    fig, axes = plt.subplots(4, 1, figsize=(21 * cm, 20 * cm), sharex=True)
    
    for i, metric_name in enumerate(["E_z", "E_xy", "S_z", "S_xy"]):
        ax = axes[i]
        data = [metrics[subtask][metric_name] for subtask in metrics]
        bplot = ax.boxplot(list(np.log10(data)), labels=metrics.keys(), patch_artist=True)
        
        # Apply styling
        for patch, median in zip(bplot['boxes'], bplot['medians']):
            patch.set_facecolor("#c9daf8")
            median.set_color("black")
        
        if metric_name in ["E_z", "E_xy"]:
            ax.set_yticks(np.arange(-2, 4))
            ax.set_yticklabels(["10$^{-2}$", "10$^{-1}$", "10$^{0}$", "10$^{1}$", "10$^{2}$", "10$^{3}$"])
            unit = "N$^2$"
        else:
            ax.set_yticks(np.arange(-3, 1))
            ax.set_yticklabels(["10$^{-3}$", "10$^{-2}$", "10$^{-1}$", "10$^{0}$"])
            unit = "N"
        
        ax.set_ylabel("$" + metric_name[0:2] + "{" + metric_name[2:] + "}$ [" + unit + "]", fontsize=8)
        ax.grid(alpha=0.3)
    
    # Set xtick labels only on bottom subplot
    axes[-1].set_xticks(np.arange(1, len(metrics) + 1))
    axes[-1].set_xticklabels(metrics.keys(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    boxplot_path = os.path.join(benchmark_data_path, "metrics_boxplot_all.pdf")
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    rosbag_force_sensor_topic = "/ftn_axia"
    benchmark_data_path = r"G:\000_LABIT_Benchmark\baseline"
    metrics_file = os.path.join(benchmark_data_path, f"qbit_metrics.npy")
    
    # read all rosbags in the benchmark folder and compute metrics
    process_trial_folders(benchmark_data_path, metrics_file)

    # plot the metrics using the saved file form above
    plot_qbit_metrics(benchmark_data_path, metrics_file)
    