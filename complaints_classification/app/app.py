from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', display="static\static daily count.png")
    else:
        csvfile = request.files['csvfile']
        inputfile = "./input_files/" + csvfile.filename
        csvfile.save(inputfile)
        with open('lgr_model.pkl', 'rb') as file:
                model = pickle.load(file)
        path = "static\model_predictions_report.png"
        deploy_plot(inputfile, model, path)
        return render_template('index.html', display=path)
    
def deploy_plot(inputfile, model, path):
        test_input = pd.read_csv(inputfile)
        test_input = pd.DataFrame(test_input)
        test_input_variables = test_input['description']
        predictions = model.predict(test_input_variables)
        test_input['predictions'] = predictions
        test_input['predictions'] = test_input['predictions'].replace(
                                        [0, 1, 2, 3, 4], 
                                        ['Billings', 'Internet Problems', 'Poor Customer Service', 'Data Caps', 'Other'])
        test_input.reset_index(drop=True)
        filename = 'predictions.csv'
        test_input.to_csv (filename, index = False, header=True)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        sns.countplot(
                x='predictions',
                data=test_input,
                dodge=False,
                hue= 'predictions', ax = ax[0])
        ax[0].set_title('Count of Predicted Categories', fontsize=16)
        ax[0].set_xticks([])
        for p in ax[0].patches:
                ax[0].annotate(f'{p.get_height()}', (p.get_x()+0.3, p.get_height()), ha='center', va='top', color='white', size=10)
        ax[0].legend(loc='best')
        grouped_by_data = test_input.groupby(['date', 'predictions'])['predictions'].count()
        grouped_by_data = pd.DataFrame(grouped_by_data)
        grouped_by_data.rename(columns = {'predictions':'number of complaints'}, inplace=True)
        grouped_by_data.reset_index(inplace=True)
        sns.lineplot(data=grouped_by_data.drop(['predictions'], axis = 1),
                x='date',  y='number of complaints', color = 'blue', linewidth=2.5, ci=None, ax = ax[1])
        dates = grouped_by_data.date.unique()
        ax[1].set_xticks([dates[0], dates[len(dates)//2], dates[-1]])
        ax[1].set_xticklabels([dates[0], dates[len(dates)//2], dates[-1]])
        ax[1].set_title('Daily Count of Complaints', fontsize=16)
        fig.suptitle('Model Predictions Report', fontsize=20, fontweight='bold')
        plt.savefig(path)
        