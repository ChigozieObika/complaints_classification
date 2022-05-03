import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

class EdaDf():
    def __init__(self, df) -> None:
        self.df = df
    
    def group_by(self, count_by, *args):
        grouped_by_state_only_data = self.df.groupby(list(args))[count_by].count()
        grouped_by_state_only_data = pd.DataFrame(grouped_by_state_only_data)
        return grouped_by_state_only_data
    

class EdaDfPlotReady():

    TOP_STATES_LIST = config.TOP_STATES_LIST
    INDEX_TOP_STATES = config.INDEX_TOP_STATES

    def __init__(self, group_by) -> None:
        self.group_by = group_by

    def date_plot(self):
        self.group_by.rename(columns = {'category':'count_by_date'}, inplace=True)
        self.group_by.reset_index(inplace=True)
        return self.group_by
    
    def category_plot(self):
        self.group_by.rename(columns = {'category':'count_by_state'}, inplace=True)
        self.group_by.sort_values(by='count_by_state', ascending=False, inplace = True)
        return self.group_by
    
    def state_category_plot(self):
        self.group_by.rename(columns = {'state':'count_by_state'}, inplace=True)
        self.group_by.reset_index(inplace=True)
        return self.group_by
    
    def top_states_count_df(self, category = None):
        if category is not None:
            top_states_df = self.state_category_plot()
            top_states_df = top_states_df[top_states_df['category']== category]
            top_states_df.sort_values(by='count_by_state', ascending=False, inplace = True)
            top_states_df.reset_index(inplace=True, drop = True)
            top_states_df = top_states_df.drop(['category'], axis = 1)
        else:
            top_states_df = self.category_plot()
        top_states_list = []
        index_top_states = [len(top_states_df)]
        index_top_states.extend(self.INDEX_TOP_STATES)
        for i in index_top_states:
            state_count = top_states_df.count_by_state[:i].sum()
            top_states_list.append(state_count)
        top_states_df = pd.DataFrame(list(zip(self.TOP_STATES_LIST, top_states_list)),
                        columns =['state_group', 'count_of_category'])
        top_states_df['percent_count'] = top_states_df['count_of_category'].apply(
                                        lambda value: round((value/top_states_df['count_of_category'][0])*100), 2)
        return top_states_df

class Plot():
    def __init__(self, df) -> None:
        self.df = df

    def plot_percent_count_top_states(self, category):
        plt.figure(figsize=(5,3), dpi=100)
        sns.barplot(data=self.df, x = 'state_group', y = 'percent_count', hue = 'state_group', dodge=False, errwidth=0)
        plt.title(f'Percentage Count of {category} by States', fontsize=12, fontweight='bold')
        plt.ylabel('Percentage Count', fontsize=10, fontweight='bold')
        plt.xlabel('State_Groups', fontsize=10, fontweight='bold')
        plt.legend(loc = 2, bbox_to_anchor = (1,1))
        plt.savefig(f'Percentage Count of {category} by States.png')
        plt.show()

    def plot_category_trend(self):
        _, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
        sns.lineplot(data=self.df.drop(['category'], axis = 1),
            x='date',  y='count_by_date', color = 'blue', linewidth=2.5, ax = ax[0,0], ci=None, )
        ax[0,0].set_title('Count of Complaints by Date - Five Categories', fontsize=14, fontweight='bold')
        sns.lineplot(data=self.df[self.df['category'] == 'billings'],
            x='date',  y='count_by_date', color = 'green', linewidth=2.5, ax = ax[0,1])
        ax[0,1].set_title('Count of Complaints by Date - Billings', fontsize=14, fontweight='bold')
        ax[0,1].set_ylim(0, 120)
        sns.lineplot(data=self.df[self.df['category'] == 'poor_customer_service'],
            x='date',  y='count_by_date', color = 'red', linewidth=2.5, ax = ax[1,0])
        ax[1,0].set_title('Count of Complaints by Date - Poor Customer Servcies', fontsize=14, fontweight='bold')
        ax[1,0].set_ylim(0, 100)
        sns.lineplot(data=self.df[self.df['category'] == 'data_caps'],
            x='date',  y='count_by_date', color = 'purple', linewidth=2.5, ax = ax[1,1])
        ax[1,1].set_title('Count of Complaints by Date - Data Caps', fontsize=14, fontweight='bold')
        ax[1,1].set_ylim(0, 100)
        sns.lineplot(data=self.df[self.df['category'] == 'internet_problems'],
            x='date',  y='count_by_date', color = 'grey', linewidth=2.5, ax = ax[2,0])
        ax[2,0].set_title('Count of Complaints by Date - Internet Probelms', fontsize=14, fontweight='bold')
        ax[2,0].set_ylim(0, 100)

        sns.lineplot(data=self.df[self.df['category'] == 'other'],
            x='date',  y='count_by_date', linewidth=2.5, ax = ax[2,1])
        ax[2,1].set_title('Count of Complaints by Date - Other', fontsize=14, fontweight='bold')
        ax[2,1].set_ylim(0, 100)

        plt.show()
            