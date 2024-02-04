import pandas as pd
import json
import seaborn as sns
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from urllib.request import urlopen
import matplotlib.pyplot as plt
import os
import numpy as np
import time

def get_data_from_url(url, data_key='values'):
    time.sleep(5)
    response = urlopen(url)
    data = json.loads(response.read())
    df = pd.DataFrame.from_dict(data[data_key])
    if 'fields' in data.keys():
        df.columns = data['fields']
    return df

def get_stages(event_id, rally_id):
    stage_url = f"https://api.wrc.com/content/result/stages?eventId={event_id}&rallyId={rally_id}&championship=wrc"
    stage_df = get_data_from_url(stage_url)
    return stage_df

def get_calendar(year):
    rally_url = f"https://api.wrc.com/content/filters/calendar?championship=wrc&origin=vcms&year={year}"
    rally_df = get_data_from_url(rally_url, data_key='content')
    return rally_df

def get_split_results(event_id, stage_id):
    split_url = f"https://api.wrc.com/content/result/splitTime?eventId={event_id}&stageId={stage_id}&championship=wrc"
    split_df = get_data_from_url(split_url)
    return split_df

def get_stage_results(event_id, rally_id, stage_id):
    stage_url = f"https://api.wrc.com/content/result/stageResult?eventId={event_id}&rallyId={rally_id}&stageId={stage_id}&championship=wrc"
    stage_df = get_data_from_url(stage_url)
    return stage_df


# try extracting all stage results from rally monte carlo 2024
    
team_colors = {'Ford' : 'white',
               'Hyundai' : 'dodgerblue',
               'Toyota' : 'firebrick',
               'Citroen' : 'darkred',
               'Skoda' : 'darkgreen',
               'Volkswagen' : 'navy',
               'Renault' : 'gold',
               'Alpine' : 'royalblue',
               'Peugeot' : 'silver',}

plotfolder = r"C:\Python\wrc_data_analysis\test_plots"

year = 2024
rally = 1

plotfolder = os.path.join(plotfolder, str(year)+'_'+str(rally))

if not os.path.exists(plotfolder):
    os.makedirs(plotfolder)

rally_df = get_calendar(year)
des_rally = rally_df[rally_df.guid == f'WRC_{year}_{rally:02}'].iloc[0]
stages_df = get_stages(des_rally.eventId, des_rally.rallyId)

des_stages = stages_df[~stages_df['STAGE'].isin(['SHD', 'FINAL'])]

all_stage_res = []
for i, row in des_stages.iterrows():
    stage_res = get_stage_results(des_rally.eventId, des_rally.rallyId, row.stageId)
    stage_res['stage_id'] = i
    if len(all_stage_res) == 0:
        all_stage_res = stage_res
    else:
        all_stage_res = pd.concat([all_stage_res, stage_res])


test = all_stage_res[all_stage_res.groupClass == 'RC1'].copy(True)

test['cleanTime'] = ['0:'+x if len(x.split(':')) < 3 else x for x in test['totalTime']]
test['convTime'] = [float(x.split(':')[0])*3600+float(x.split(':')[1])*60+float(x.split(':')[2]) for x in test['cleanTime']]

# ax = sns.lineplot(data=test, x='stage_id', y='convTime', hue='driver')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# seaborn version
ax = sns.lineplot(data=test, x='stage_id', y='convTime', hue='driver')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

# plotly version

all_stage_res['cleanTime'] = ['0:'+x if len(x.split(':')) < 3 else x for x in all_stage_res['totalTime']]
all_stage_res['convTime'] = [float(x.split(':')[0])*3600+float(x.split(':')[1])*60+float(x.split(':')[2]) for x in all_stage_res['cleanTime']]

app = Dash(__name__)


app.layout = html.Div([
    html.H4('Life expentancy progression of countries per continents'),
    dcc.Graph(id="graph"),
    dcc.Checklist(
        id="checklist",
        options=["RC1", "RC2", "RC3"],
        value=["RC1"],
        inline=False
    ),
])


@app.callback(
    Output("graph", "figure"), 
    Input("checklist", "value"))
def update_line_chart(cur_class):
    df = all_stage_res # replace with your own data source
    mask = df.groupClass.isin(cur_class)
    fig = px.line(df[mask], 
        x="stage_id", y="convTime", color='driver')
    return fig


app.run_server(debug=False)




# try extracting split times per stage and normalizing to winner
sns.set_style("darkgrid", {'grid.linestyle': '--', 
                           'axes.facecolor' :'dimgray',
                           'axes.edgecolor' : 'black',
                           'grid.color' : 'darkgray'})
# dash_cycle = [(0, ()), (0, (5, 5)), (0, (1, 1)), (0, (3, 5, 1, 5))]
dash_cycle = ['-', '--', ':', '-.']

# marker_cycle = ['o', 's', 'p']

# plot split times for each stage and each car class
for i, row in des_stages.iterrows():
    split_df = get_split_results(des_rally.eventId, row.stageId)    
    org_splits = [x for x in split_df.columns if 'round' in x] + ['diffFirst']
    for split in org_splits:
        split_df.loc[0, split] = '+0.0'

    for car_class in split_df.groupClass.unique():

        temp_split = split_df[split_df.groupClass == car_class].copy(deep=True)
        all_splits = [x for x in temp_split.columns if 'round' in x] + ['diffFirst']
        
        for entry in all_splits:
            split_times = [x.split(':') if type(x) == str else np.nan for x in temp_split[entry]]
            new_split_times = []
            for time in split_times:
                if type(time) != list:
                    new_split_times.append(np.nan)
                else:
                    if len(time) > 1:
                        if len(time) == 2:
                            new_split_times.append(float(time[0])*60+float(time[1]))
                        elif len(time) == 3:
                            new_split_times.append(float(time[0])*3600+float(time[1])*60+float(time[2]))
                    else:
                        if time[0] == '':
                            new_split_times.append(0.0)
                        else:
                            new_split_times.append(float(time[0]))

            temp_split[entry] = new_split_times

        temp_split = temp_split.sort_values('diffFirst', ascending=True)
        for entry in all_splits:
            temp_split[entry] = temp_split[entry] - temp_split.iloc[0][entry]

        temp_split['round0'] = 0
        all_splits = ['round0'] + all_splits
        plot_view = temp_split[['driver', 'teamName'] + all_splits].melt(id_vars=['driver', 'teamName'], var_name='split', value_name='time')
        plot_view['split_num'] = plot_view.split.map({x : i for i,x in enumerate(all_splits)})
        
        dash_dict = {}
        for team in plot_view.teamName.unique():
            for d, driver in enumerate(plot_view[plot_view.teamName == team].driver.unique()):
                dash_dict[driver] = dash_cycle[d%3]



        fig, axs = plt.subplots()
        for team in plot_view.teamName.unique():
            for d, driver in enumerate(plot_view[plot_view.teamName == team].driver.unique()):
                temp_data = plot_view[(plot_view.teamName == team) & 
                                      (plot_view.driver == driver)]
                
                axs.plot(temp_data.split_num, 
                         temp_data.time, 
                         color=team_colors[team], 
                         linestyle=dash_dict[driver],
                         lw=1.5,
                         label=driver)
                

        axs.invert_yaxis()
        axs.set_xticks(temp_data.split_num)
        axs.set_xticklabels('')
        axs.set_ylabel('Time to leader [s]')
        axs.set_title(row['name']+ '\n' + car_class)
        # check for distribution of points and adjust xlim to last point in 2*std of mean
        end_times = [line.get_ydata()[-1] for line in axs.lines]
        mean = np.mean(end_times)
        std = np.std(end_times)

        if axs.get_ylim()[0] >  mean + 1*std:
            max(np.array(end_times)[end_times < mean + 1*std])
            axs.set_ylim(max(np.array(end_times)[end_times < mean + 1*std])+2 , axs.get_ylim()[1])
        
       

        axs.set_xticks(temp_data.split_num[:-1].values+0.5, minor=True)
        axs.set_xticklabels(['Split '+str(i) for i in np.arange(len(temp_data.split_num[:-1]))], 
                           minor=True)
        axs.tick_params(axis='x', which="minor",length=0)

        lgd = axs.legend(loc='upper center', ncols=3, bbox_to_anchor=(0.5, -0.1), 
                         facecolor='none', edgecolor='none')
        fig.set_facecolor('lightgray')
        fig.savefig(os.path.join(plotfolder, row.STAGE + '_' + car_class + '.png'),
                    bbox_inches='tight')
        plt.close()
        # plt.savefig(os.path.join(plotfolder, row.STAGE + '_' + car_class + '.png'))
        # plt.close()
