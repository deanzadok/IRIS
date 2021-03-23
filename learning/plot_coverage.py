import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_dir, trials_num=300):

    columns = np.round(np.arange(0.1, 1.1, 0.1),1).tolist()
    coverage_df = pd.DataFrame(columns=columns)
    trials_counter = 0

    for i in range(trials_num):

        row_dict = {}

        for j in columns:

            files_list = {'obstacles': os.path.join(data_dir, f'test_p_zb_{i}_{j}_obstacles.csv'),
                        'inspection_points': os.path.join(data_dir, f'test_p_zb_{i}_{j}_inspection_points.csv'),
                        'configurations': os.path.join(data_dir, f'test_p_zb_{i}_{j}_conf'),
                        'vertex': os.path.join(data_dir, f'test_p_zb_{i}_{j}_vertex'),
                        'results_full': os.path.join(data_dir, f'test_search_p_zb_{i}_{j}'),
                        'results':os.path.join(data_dir, f'test_search_p_zb_{i}_{j}_result')}

            # check if test files exist
            broken_files = False
            for file_path in files_list.values():
                if not os.path.isfile(file_path):
                    broken_files = True
            
            if broken_files:
                continue

            coverages = pd.read_csv(files_list['results_full'], header=None, sep=' ')[6]
            row_dict[j] = coverages.iloc[-1]
            
            trials_counter += 1

        coverage_df = coverage_df.append(row_dict, ignore_index=True)

    return coverage_df, columns, trials_counter


def plot_coverage_means(coverage_df, xs, counter):

    coverage_means = coverage_df.mean().values
    coverage_stds = coverage_df.std().values
    stes = 1.96 * coverage_stds / np.sqrt(counter)

    # plor results
    plt.plot(xs, coverage_means)
    plt.fill_between(xs, coverage_means + stes, coverage_means - stes, alpha=0.5)
    plt.legend(['Coverage'])
    plt.xlabel('Probability')
    plt.ylabel('Coverage')
    plt.title('Coverage per probability')
    plt.grid(True)
    plt.savefig("coverage_p.png")
    plt.clf()

def plot_coverage_single(coverages, xs, idx):

    # plor results
    plt.plot(xs, coverages)
    plt.legend(['Coverage'])
    plt.xlabel('Probability')
    plt.ylabel('Coverage')
    plt.title('Coverage per probability')
    plt.grid(True)
    plt.savefig(f'coverage_{idx}.png')
    plt.clf()

def plot_coverage_hist(coverages_sums, xs):

    # plor results
    plt.bar(xs, coverages_sums)
    plt.xlabel('Probability')
    plt.ylabel('Coverage histogram')
    plt.title('Coverage histogram')
    plt.grid(True)
    plt.savefig('coverage_hist.png')
    plt.clf()


if __name__ == '__main__':

    data_dir = 'build/test_fixed_p_zb'
    coverage_df, xs, counter = load_data(data_dir)
    unwanted_columns = []
    for xi in xs:
        unwanted_columns.append(f'mt{xi}')
        coverage_df[f'mt{xi}'] = coverage_df[xs].apply(lambda x: 1 if x.max() > x[xi] else 0, axis=1)
    selected_coverage_df = coverage_df[(coverage_df['mt0.8'] == 1) & (coverage_df['mt0.9'] == 1) & (coverage_df['mt1.0'] == 1)]

    plot_coverage_means(coverage_df[xs], xs, counter)

    for xi in xs:
        coverage_df[f'e{xi}'] = coverage_df[xs].apply(lambda x: 1 if x.max() == x[xi] else 0, axis=1)
    
    plot_coverage_hist(coverage_df.drop(columns=xs+unwanted_columns).sum().values.tolist(), [str(x) for x in xs])