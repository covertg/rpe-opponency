import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import sys

import worlds

TINYNONZERO = 10 ** sys.float_info.min_10_exp

def softmax(X, β=1):
    '''
    Softmax function with inverse-temperature "β" parameter.
    Hopefully this is numerically stable; see https://cs231n.github.io/linear-classify/#softmax.
    '''
    assert not np.isnan(X).any()
    exp_vals = np.exp(β * (X - np.max(X)))
    return exp_vals / np.sum(exp_vals)


def entropy(X, min, max, σ_noise, bins=200):
    '''
    Calculates entropy given a vector of values assumed to come from a range [xmin, xmax] but with extra noise applied.
    
    :param X: Vector of values of type numpy.ndarray with a float dtype
    :param min: Lower bound on X's elements without noise
    :param max: Upper bound on X's elements without noise
    :param σ_noise: Standard deviation of the noise applied to x
    :param bins: Number of bins to use when calculating the counts of unique values (default 200)
    '''
    assert not np.isnan(X).any()
    assert X.dtype == np.float or X.dtype == np.float32 or X.dtype == np.float64
    if len(X) <= 1:
        return 0
    clip = (min - σ_noise, max + σ_noise)
    Xclip = np.clip(X, *clip)
    counts, _ = np.histogram(Xclip, np.linspace(*clip, bins))  # 'counts' as an unnormalized PMF
    entropy = scipy.stats.entropy(counts, base=2)  # scipy handles zero-probability cases nicely
    if np.isnan(entropy).any():
        print('nan entropy from vec', X)
    return entropy


def shift_reward_integration(orig_weight_past, orig_weight_pres, past_shift):
    new_weight_past = orig_weight_past + past_shift
    new_weight_pres = orig_weight_pres * (1 - new_weight_past) / (1 - orig_weight_past)
    return {'past': new_weight_past, 'pres': new_weight_pres}


def get_SAD(W, A, n_states, title=None, rowheight=3):
    '''
    Plots single-agent dashboard with seaborn/matplotlib.
    '''
    sns.set(rc={'figure.figsize': (16, rowheight * (n_states + 2)), 'figure.dpi': 300})
    sns.set_style('white')
    sns.set_palette('colorblind')
    states2plot = range(n_states)
    trials = range(W.trial)
    f, axs = plt.subplots(2 + len(states2plot), 4)
    if title:
        f.suptitle(title)

    # ROW 1: state-choice info and MI
    #max_state = [np.max(A.memory.states[:j+1]) for j in trials]
    state_selected = A.memory.states
    # states = pd.DataFrame({'state choice': state_selected, 'trial': trials}).melt(id_vars='trial', value_name='state')
    # p00 = sns.scatterplot(data=states, x='trial', y='state', hue='variable', ax=axs[0, 0], linewidth=0, s=8)
    p00 = sns.scatterplot(x=trials, y=state_selected, ax=axs[0, 0], linewidth=0, s=8)
    p00.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    p00.set_xlabel('trial')
    p00.set_ylabel('state selected')
    p00.set_title('agent state inference')

    min_surprise = [np.min(F) for F in A.memory.logs['surprise'][1:-1]]  # Slice the first trial away because surprise is empty
    # max_surprise = [np.max(F) for F in A.memory.logs['surprise'][1:]]
    # avg_surprise = [np.mean(F) for F in A.memory.logs['surprise'][1:]] 
    # surprise = pd.DataFrame({'minimum': min_surprise, 'trial': trials[1:]}).melt(id_vars='trial', var_name='metric', value_name='surprise')
    # p01 = sns.scatterplot(data=surprise, x='trial', y='surprise', hue='metric', ax=axs[0, 1], linewidth=0, s=8)
    p01 = sns.scatterplot(x=trials[1:], y=min_surprise, ax=axs[0, 1], linewidth=0, s=8)
    p01.set_xlabel('trial')
    p01.set_ylabel('minimum surprise $F$')
    p01.set_title('minimum surprise over all agent-states')

    I_SC = pd.DataFrame(A.memory.logs['I_SC']).join(pd.Series(trials, name='trial')).melt(id_vars='trial', var_name='cue', value_name='mutual information')
    I_SC['cue'] = I_SC['cue'].apply(str)
    p02 = sns.lineplot(data=I_SC, x='trial', y='mutual information', hue='cue', ax=axs[0, 2])
    p02.set_title('state-comparison cue information ($I_{SC}$)')

    cueweight = pd.DataFrame(A.memory.logs['cueweight'])
    cueweight.columns = [f'cue {c} ($w_A[{c}]$)' for c in cueweight.columns]
    cueweight = cueweight.join(pd.Series(A.memory.logs['dbscale'], name='$\\bar \\delta$-scaled')).\
        join(pd.Series(trials, name='trial')).\
        melt(id_vars='trial', var_name='weight', value_name='cue-related weight')
    p03 = sns.scatterplot(data=cueweight, x='trial', y='cue-related weight', hue='weight', ax=axs[0, 3], linewidth=0, s=8)
    p03.set_title('cue-related weights ($w_A, \\bar \\delta$-scaled)')

    # ROW 2: action and reward/RPE-related info
    actionrewards = pd.DataFrame({'action choice': A.memory.actions, 'reward': A.memory.rewards, 'trial': trials})
    p10 = sns.scatterplot(data=actionrewards, x='trial', y='action choice', hue='reward', ax=axs[1, 0], linewidth=0, s=8)
    axs[1, 0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    p10.set_title('action choice and reward')

    best_action = W.logs['best_action']
    best_action_prob = [A.memory.logs['action_prob'][trial, a] for trial, a in enumerate(best_action)]
    p11 = sns.scatterplot(x=trials, y=best_action_prob, ax=axs[1, 1], linewidth=0, s=8)
    p11.set_xlabel('trial')
    p11.set_ylabel('p(best action chosen)')
    p11.set_title('agent performance by action probability')

    reward_errs = pd.DataFrame(A.memory.logs['reward_integration'])
    if '$\\bar r$' in reward_errs.columns:
        rpe = reward_errs[['$\\delta_t$', '$\\bar r$']]
        rpe = rpe.join(pd.Series(trials, name='trial'))
        dbars = reward_errs.drop(columns=['$\\delta_t$', '$\\bar r$'])
    else:
        rpe = reward_errs['$\\delta_t$']
        rpe = pd.concat({'$\\delta_t$': rpe, 'trial': pd.Series(trials)}, axis=1)
        dbars = reward_errs.drop(columns='$\\delta_t$')

    rpe = rpe.melt(id_vars='trial', var_name='prediction', value_name='value')
    p12 = sns.scatterplot(data=rpe, x='trial', y='value', hue='prediction', ax=axs[1, 2], linewidth=0, s=8)
    p12.set_title('reward prediction')

    dbars = dbars.join(pd.Series(trials, name='trial')).melt(id_vars='trial', var_name='signal', value_name='value')    
    p13 = sns.scatterplot(data=dbars, x='trial', y='value', hue='signal', ax=axs[1, 3], linewidth=0, s=8)
    p13.set_title('reward integration')

    # ROW 3 and beyond: action-values and prototypes for each state
    for i in states2plot:
        first_trial_i = np.flatnonzero(A.memory.states == i)[0]
        trials_i = range(first_trial_i, W.trial + 1)

        action_val_i = [Q[i] for Q in A.memory.logs['action_val'][first_trial_i:]]
        action_val_i_df = pd.DataFrame(action_val_i).join(pd.Series(trials_i, name='trial')).melt(id_vars='trial', var_name='action', value_name='value')
        px0 = sns.lineplot(data=action_val_i_df, x='trial', y='value', hue='action', ax=axs[2+i, 0])
        axs[2+i, 0].set_xlim((0, W.trial))
        px0.set_title(f'action-values for $s_{i}$ ($Q[{i}]$)')

        proto_mean_i = [mus[i] for mus in A.memory.logs['proto_mean'][first_trial_i:]]
        proto_mean_i_df = pd.DataFrame(proto_mean_i).join(pd.Series(trials_i, name='trial')).melt(id_vars='trial', var_name='cue', value_name='mean')
        proto_mean_i_df['cue'] = proto_mean_i_df['cue'].apply(str)
        px1 = sns.lineplot(data=proto_mean_i_df, x='trial', y='mean', hue='cue', ax=axs[2+i, 1])
        axs[2+i, 1].set_xlim((0, W.trial))
        px1.set_title(f'prototype means for $s_{i}$ ($\\mu_{i})$')

        I_SD_is = [I_SD[i] for I_SD in A.memory.logs['I_SD'][first_trial_i:]]
        I_diag = [np.diag(I_SD_i) for I_SD_i in I_SD_is]
        I_diag_df = pd.DataFrame(I_diag).join(pd.Series(trials_i, name='trial')).melt(id_vars='trial', var_name='cue', value_name='precision weight')
        I_diag_df['cue'] = I_diag_df['cue'].apply(str)
        px2 = sns.lineplot(data=I_diag_df, x='trial', y='precision weight', hue='cue', ax=axs[2+i, 2])
        axs[2+i, 2].set_xlim((0, W.trial))
        px2.set_title(f'diagonal of $I_{{SD,{i}}}$')

        first_trial_i_surprise = max(1, first_trial_i + 1)
        surprise_i = [F[i] for F in A.memory.logs['surprise'][first_trial_i_surprise:-1]]
        surprisetrials = range(first_trial_i_surprise, W.trial)
        px3 = sns.scatterplot(x=surprisetrials, y=surprise_i, ax=axs[2+i, 3], linewidth=0, s=8)
        axs[2+i, 3].set_xlim((0, W.trial))
        px3.set_xlabel('trial')
        px3.set_ylabel('surprise')
        px3.set_title(f'surprise $F$ for $s_{i}$')
    
    # Minimal vertical lines on all subplots to identify each block
    if isinstance(W, worlds.RewardBlockWorld):
        vlines_x = np.cumsum(W.P['blocks']['block_lengths'][:-1])
        for r in range(len(axs)):
            for c in range(len(axs[r])):
                ax = axs[r, c]
                for x in vlines_x:
                    ax.axvline(x=x, color='black', alpha=0.5, linestyle='dotted')

    f.tight_layout()
    return f