import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def shuffle_and_sample(x, txt_path):
    shuffled = x.sample(frac = 1, random_state = 0) # Shuffle the data with a fixed random state
    # We repeat the eval on the test set 5 times, but 9 times on the validation set
    top_n = 5 if 'test' in txt_path else 9
    return shuffled.head(top_n)
def remove_underscore_after(val): # Remove underscore, and keep the part after the underscore
    return val.split('_')[-1]
def remove_underscore_before(val): # Remove underscore, and keep the part before the underscore
    return val.split('_')[0]
def map2d(func, grid): # Mapping for 2d arrays, from: https://stackoverflow.com/questions/70742445/elegant-map-over-2d-list
    return [[func(value) for value in row] for row in grid]
def full_display(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)
def prep_and_store_results(txt_path: str): # Prepare the (text-file) results to be stored in a csv
    with open(txt_path, 'r') as f:
        results = f.readlines()

    # Remove any non-result lines from the eval file, and split the lines on the tab character
    # (results have format: model_name\tdataset_name\tmetric_name\tmetric_value)
    results = [r.replace('\n','').split('\t') for r in results if '\t' in r]

    # Make a dataframe from the results
    df = pd.DataFrame(results, columns = ['model', 'dataset', 'metric', 'value'])
    df['value'] = pd.to_numeric(df['value'])
    # Remove the timestamp from the model names
    df['model'] = df['model'].map(lambda x: '-'.join(x.split('-')[2:]))

    # Make a list of model names, split by parameters - model names look like var1_xxx-var2_yyy-var3_zzz-... so split on '-'
    models = df['model'].str.split('-').tolist()
    
    # Remove all underscores from our 2d list, keep one list of the param names and one with param vals
    model_names_list = map2d(remove_underscore_before, models) # Keep the part before the underscore, aka the variable name
    model_names = np.array(model_names_list)
    model_val_list = map2d(remove_underscore_after, models) # Keep the part after the underscore, aka the variable's value
    model_vals = np.array(model_val_list)
    
    print('Number of evaluations:', model_vals.shape[0])

    # Splitting model name into columns, using the list of variables and their values
    for i in range(model_names.shape[-1]):
        name = model_names[0][i]
        val = model_vals[:,i]
        df[name] = val
        try: # Try to make columns numeric if possible
            df[name] = pd.to_numeric(df[name]) 
        except:
            pass
    
    # Remove columns that aren't used
    df = df.drop(['vit', 'model', 'data',   'kw'], axis = 1) #'method', 'AL.iter', 'ratio', 'PL', 'ALL',
    if 'fold' in df.columns.tolist():
        df = df.drop(['fold'], axis = 1)
    
    # Replace 'None' with NaN, to allow conversion to numerical
    df['AL.iter'] = df['AL.iter'].replace('None', np.nan)
    df['AL.iter'] = pd.to_numeric(df['AL.iter'])
    df['AL.epochs'] = df['AL.epochs'].replace('None', np.nan)
    df['AL.epochs'] = pd.to_numeric(df['AL.epochs'])

    cols = sorted(df.columns.tolist()) # Get a list of the columns of the dataframe
    print('Column names:', cols)

    display(df)

    # Group by the model parameters 
    df_grouped = df.groupby(list(set(cols)-set(['value'])), dropna = False)#.sample(frac=1).head(5 if 'test' in txt_path else 9)
    display(df_grouped.head(5 if 'test' in txt_path else 9))
    # Compute mean, std, max, min performance and number of runs for each model 
    df_grouped = df_grouped.agg({ # randomly pick X model runs to use in the analysis (5 for test, 9 for val) 
        'value': [('mean', lambda x: shuffle_and_sample(x, txt_path).mean()), 
                    ('std', lambda x: shuffle_and_sample(x, txt_path).std()), 
                    ('count', lambda x: shuffle_and_sample(x, txt_path).shape[0]),
                    ('all', lambda x: list(shuffle_and_sample(x, txt_path))), # keep a list of all the runs
                    ('min', lambda x: shuffle_and_sample(x, txt_path).min()),
                    ('max', lambda x: shuffle_and_sample(x, txt_path).max()),     
                 ]
    }) 

    df_grouped.to_csv(txt_path.replace('.txt', '.csv'))
    display(df_grouped)
    return df_grouped

def get_results_per_method(df, hyperparam_tuning = True):
    if hyperparam_tuning: # Only report on the results for a specific label ratio if we're hyperparam tuning
        df = df[(df['ratio'] == 0.1)]
#     df_no_finetune = df[(df['epochs']==0)]
    df_baseline = df[((df['AL.iter'].isna()) & (df['method'] == 'base') & (df['epochs'] > 0)) | (df['ratio'] == 0 )]
    df_S_CLIP = df[(df['AL.iter'].isna()) & (df['method'] == 'ours') & (df['PL'].str.contains('ot.'))]
    df_soft_PL = df[(df['AL.iter'].isna()) & (df['method'] == 'ours') & (df['PL'].str.contains('soft.'))]
    df_hard_PL = df[(df['AL.iter'].isna()) & (df['method'] == 'ours') & (df['PL'].str.contains('hard.'))]
    df_basic_AL = df[(df['AL.iter']>=0) & (df['ProbVLM']=='False') & (df['AL.epochs']<=20)]
    df_probvlm_AL = df[(df['ProbVLM']=='True')]
    
    return { # return a dictionary of results per model
        'baseline': df_baseline, 's-clip': df_S_CLIP,  #'baseline-not-finetuned' : df_no_finetune, 
        'soft-pl': df_soft_PL, 'hard-pl': df_hard_PL, 'basic-al': df_basic_AL, 'probvlm': df_probvlm_AL 
    }
    
def performance_per_label_ratio(df, metric, dataset):
    # Filter for the specified dataset and metric
    df_filtered = df[(df['metric'] == metric) & (df['dataset'] == dataset)]
    
    # Ensure the order is from the smallest label ratio to the largest and display
    df_filtered = df_filtered.sort_values(by='ratio')
    
    # Get the mean, std, min, max performance on the given metric & dataset
    mean = df_filtered[('value', 'mean')].to_numpy()
    std = df_filtered[('value', 'std')].to_numpy()
    min_ = df_filtered[('value', 'min')].to_numpy()
    max_ = df_filtered[('value', 'max')].to_numpy()

    return {'mean': mean, 'std': std, 'min': min_, 'max': max_}


def plot_model_comparison(results_dict, metric, dataset, which_models = 'all', epochs_dict = {}, ax = None,
                         save_results = True, display_table_results = False, label = True, base_fontsize = 14,
                         confidence_band_type = 'min-max'):
    one_plot = False # Keep track of whether we are dealing with one or more plots
    if ax is None: # Set an axis
        fig, ax = plt.subplots(figsize=(12, 6)) 
        one_plot = True
    only_scores = {}
    models = results_dict.keys()
    
    # Filter for specific types of models
    if which_models == 'pseudo-labeling':
        models = ['baseline', 's-clip', 'soft-pl', 'hard-pl'] 
    if which_models == 'active-learning':
        models = ['baseline', 'basic-al']  # 'probvlm'
        
    # Which epochs to filter to, for each model
    if not epochs_dict: # Use default dictionary if one wasn't specified
        epochs = {'baseline': [0, 25], 'basic-al': [20], 'probvlm': [25], 
                  's-clip': [25], 'soft-pl': [30], 'hard-pl': [25], }
    
    # The label ratios that we use
    label_ratios = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0] 
    
    # Get list of colors (used to make the plot of the mean and the std around it have the same color)
    cmap = plt.cm.get_cmap("tab10")
    color_list = cmap.colors
    
    # Add information about the dataset, metric and label ratios to the plot
    metric_formatted = metric.replace('_', ' ')
    ax.set_title(f'{metric_formatted} (dataset: {dataset})', fontsize = base_fontsize + 2)
    ax.set_xticks(label_ratios, label_ratios, fontsize = base_fontsize-2, rotation = 90)
    ax.tick_params(axis='y', labelsize= base_fontsize - 2)
    ylabel = 'recall' if 'R@' in metric else 'accuracy'
    ax.set_ylabel(ylabel, fontsize = base_fontsize)
    ax.set_xlabel('Label ratio', fontsize = base_fontsize)
    
    # Get the performance of each model, for the given metric and dataset
    for i, model in enumerate(models):
        model_results = results_dict[model]
        
        # If we have any results for the given model, add it to the plot
        if model_results is not None and model_results.shape[0] > 0: 
            # Filter for correct number of epochs
            model_results = model_results[model_results['epochs'].isin(epochs[model])]
            if display_table_results:
                print(model, metric, dataset)
                display(df_filtered)
            performance = performance_per_label_ratio(model_results, metric, dataset)
            mean = performance['mean']
            
            # Pad the performance if it misses results at label_ratio = 0 (start) and label_ratio = 1 (end)
            pad_end = 1 if len(mean) != len(label_ratios) else 0
            # compute how much to pad at start - this is equal to 1 if all results are known
            pad_start = len(label_ratios) - len(mean) - pad_end 
            
            # Pad the mean, std, max and min performance, if necessary, at label_ratio = 0 and = 1
            mean = np.pad(mean, (pad_start, pad_end), 'constant', constant_values=np.nan)
            max_ = np.pad(performance['max'], (pad_start, pad_end), 'constant', constant_values=np.nan)
            min_ = np.pad(performance['min'], (pad_start, pad_end), 'constant', constant_values=np.nan)
            std = np.pad(performance['std'], (pad_start, pad_end), 'constant', constant_values=np.nan) 

            ax.plot(label_ratios, mean, label = model if label else None, linestyle = '-', color = color_list[i])

            # Add the minimum and maximum performance, or its std, as a 'band' of confidence around the mean
            if confidence_band_type == 'std':
                ax.fill_between(label_ratios, mean-std, mean+std, alpha=0.2, color = color_list[i])
            elif confidence_band_type == 'min-max' or confidence_band_type == 'max-min':
                ax.fill_between(label_ratios, min_, max_, alpha=0.2, color = color_list[i])
            # Store the list with all the scores (for the given metric & dataset) in a dictionary
            model_results_all_scores = model_results[('value', 'all')].tolist()
            only_scores[model] = model_results_all_scores
    
    # Generic path for the results of this method, metric, dataset
    file_path = f'./results/data_{dataset}_metric_{metric}_methods_{which_models}'
    if one_plot:
        ax.legend()
        # Store the image both as a pdf and a png for flexibility
        plt.savefig(file_path + '.png', transparent = True, bbox_inches='tight')
        plt.savefig(file_path + '.pdf', transparent = True, bbox_inches='tight')
        plt.show()
    if save_results:
        model_results.to_csv(file_path + '.csv')
    return only_scores

def plot_recall_model_comparison(results_dict, retrieval_metrics, retrieval_datasets, base_fontsize = 14, k = 1, 
                                confidence_band_type = 'min-max'):
    r_m, r_d = len(retrieval_metrics), len(retrieval_datasets)
    fig, axes = plt.subplots(r_m, r_d, figsize = (30, 15))
    for row, metric in enumerate(retrieval_metrics):
        for col, dataset in enumerate(retrieval_datasets):
            metric_at_k = metric.format(k)
            label = row == 0 and col == 0 # only assign a legend label for the first subplot (prevents duplicates)
            plot_model_comparison(results_dict, metric_at_k, dataset, ax = axes[row][col], label = label,
                                 base_fontsize = base_fontsize, confidence_band_type = confidence_band_type)
    fig.legend(loc='upper center', ncol=6, fancybox=True,  bbox_to_anchor=(0.5, 1.05), fontsize = base_fontsize, 
              )
    plt.tight_layout()
    plt.show()