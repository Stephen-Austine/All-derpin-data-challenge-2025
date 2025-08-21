from flask import *
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/forecast')
def forecast():
    return render_template('forecast.html')


@app.route('/selection', methods=['GET', 'POST'])
def selection():
    if request.method == 'POST':
        region = request.form["region"]
        print("Region selected:", region)
    else:
        return render_template('selection.html')
    
    return render_template('selection.html')


@app.route('/forecastselection')
def forecastselection():
    return render_template('forecastselection.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/help')
def help():
    return render_template('help.html')

app.run(debug=True)
