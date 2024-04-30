from flask import Flask, render_template, request
import datetime
import requests
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from dateutil.parser import parse as parse_date

# Initialize Flask app
app = Flask(__name__)

# Define Google LLM for interacting with Google Calendar
llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0.6, google_api_key="AIzaSyDjITo6JpwACzQKlMCJKuBhHHK8jTQIhBg")

# Define agents
farmer_agent = Agent(role='Farmer Agent', goal='Gather planting information from the farmer',
                     backstory='An agent specialized in interacting with farmers to gather planting information.',
                     verbose=True, allow_delegation=False, llm=llm)

agronomist_agent = Agent(role='Agronomist Local Expert', goal='Provide personalized farming advice based on location and crop',
                         backstory='An expert specialized in providing personalized farming advice based on location and crop.',
                         verbose=True, allow_delegation=False, llm=llm)

planner_agent = Agent(role='Amazing Planner Agent', goal='Create an optimized planting calendar with budget and best farming practices',
                      backstory='Specialist in farm management and agronomy with decades of experience, providing a calendar based on the provided information.',
                      verbose=True, allow_delegation=False, llm=llm)

crop_suggestion_agent = Agent(role='Crop Suggestion Agent', goal='Suggest alternative crops if the entered crop is out of season',
                              backstory='An agent specialized in suggesting alternative crops based on seasonality and profitability in that local area.',
                              verbose=True, allow_delegation=False, llm=llm)

# Define tasks
planting_info_task = Task(description='Gather planting information from the farmer: {plant}', agent=farmer_agent,
                          expected_output='Planting information collected from the farmer.')

farming_advice_task = Task(description='Provide personalized farming advice for {crop} in {location} starting from {start_date}.',
                           agent=agronomist_agent, expected_output='Personalized farming advice provided.')

farming_calendar_task = Task(description='Generate farming calendar for {crop} in {location} starting from {start_date}.',
                             agent=planner_agent, expected_output='Farming calendar generated.')

season_check_task = Task(description='Check if the planting season has ended for {crop} in {location} by {current_date}.',
                         agent=agronomist_agent, expected_output='Planting season status checked.')

crop_suggestion_task = Task(description='Suggest alternative crops if {crop} is out of season for {location} by {current_date}.',
                            agent=crop_suggestion_agent, expected_output='Alternative crops suggested.')

farming_itinerary_task = Task(description='Display farming itinerary for {crop} in {location} starting from {start_date}.',
                              agent=agronomist_agent, expected_output='Farming itinerary displayed.')

# Define crews
farming_crew_planting = Crew(agents=[farmer_agent, agronomist_agent, planner_agent, crop_suggestion_agent],
                              tasks=[planting_info_task, farming_advice_task, farming_calendar_task, season_check_task,
                                     crop_suggestion_task, farming_itinerary_task], verbose=True, process=Process.sequential)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/planting_calendar', methods=['GET', 'POST'])
def planting_calendar():
    if request.method == 'POST':
        location = request.form['location']
        crop = request.form['crop']
        start_date = request.form['start_date']

        try:
            # Interpolate farmer's planting information into the tasks descriptions
            planting_info_task.interpolate_inputs({"plant": crop})
            farming_advice_task_inputs = {"crop": crop, "location": location, "start_date": start_date}
            farming_advice_task.interpolate_inputs(farming_advice_task_inputs)
            farming_calendar_task_inputs = {"crop": crop, "location": location, "start_date": start_date}
            farming_calendar_task.interpolate_inputs(farming_calendar_task_inputs)
            current_date = datetime.date.today()
            season_check_task_inputs = {"crop": crop, "location": location, "current_date": current_date}
            season_check_task.interpolate_inputs(season_check_task_inputs)
            crop_suggestion_task_inputs = {"crop": crop, "location": location, "current_date": current_date}
            crop_suggestion_task.interpolate_inputs(crop_suggestion_task_inputs)
            farming_itinerary_task_inputs = {"crop": crop, "location": location, "start_date": start_date}
            farming_itinerary_task.interpolate_inputs(farming_itinerary_task_inputs)

            # Get weather information for the specified location
            openweathermap_api_key = "bb7a7d944437ed1b8df2a27b54490cbb"
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={openweathermap_api_key}&units=metric"
            weather_response = requests.get(weather_url)
            weather_data = weather_response.json()

            # Execute the farming crew for planting calendar
            result = []
            for task in farming_crew_planting.tasks:
                result.append(f"Executing task: {task.description}")
                output = task.execute()
                result.append("Task completed successfully!")
                if task.agent == agronomist_agent:
                    result.append("Agronomist's Advice:")
                    result.append(output)
                elif task.agent == planner_agent:
                    result.append("Farming Calendar:")
                    result.append(output)

            # Render result template with the task execution output
            return render_template('planting_calendar_result.html', weather_data=weather_data, result=result)

        except ValueError:
            return render_template('error.html', message="Invalid input. Please enter valid values.")

    return render_template('planting_calendar_form.html')

if __name__ == '__main__':
    app.run(debug=True)
