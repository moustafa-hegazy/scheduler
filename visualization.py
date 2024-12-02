import json
import pandas as pd
import datetime

json_file_path = '/Users/moustafahegazy/Documents/GitHub/scheduler/grad_proj/output_data/schedule_output.json'


with open(json_file_path, 'r') as f:
    schedule_data = json.load(f)

df = pd.DataFrame(schedule_data)

def parse_time_range(time_range):
    start_str, end_str = time_range.split('-')
    start_time = datetime.datetime.strptime(start_str, '%H:%M')
    end_time = datetime.datetime.strptime(end_str, '%H:%M')
    return start_time, end_time


df['start_time'] = df['time'].apply(lambda x: parse_time_range(x)[0])
df['end_time'] = df['time'].apply(lambda x: parse_time_range(x)[1])


day_order = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
df['day_order'] = df['day'].map(day_order)
df = df.sort_values(['day_order', 'start_time'])
grouped = df.groupby('day')

html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>University Schedule</title>
    <style>
        body {font-family: Arial, sans-serif; margin: 20px;}
        h2 {background-color: #004080; color: white; padding: 10px;}
        table {width: 100%; border-collapse: collapse; margin-bottom: 30px;}
        th, td {border: 1px solid #dddddd; padding: 8px; text-align: left; vertical-align: top;}
        th {background-color: #f2f2f2;}
        tr:nth-child(even) {background-color: #fafafa;}
        .instructor {font-style: italic; color: #555;}
    </style>
</head>
<body>
    <h1>Weekly Schedule</h1>
'''


for day, day_df in grouped:
    html_content += f'<h2>{day}</h2>\n'
    html_content += '''
    <table>
        <tr>
            <th>Time</th>
            <th>Course Code</th>
            <th>Course Name</th>
            <th>Session Type</th>
            <th>Group</th>
            <th>Room</th>
            <th>Instructor</th>
        </tr>
    '''
    for _, row in day_df.iterrows():
        html_content += f'''
        <tr>
            <td>{row["time"]}</td>
            <td>{row["course_code"]}</td>
            <td>{row["course_name"]}</td>
            <td>{row["session_type"]}</td>
            <td>{row["group"]}</td>
            <td>{row["room"]}</td>
            <td>{row["instructor"]} ({row["instructor_type"]})</td>
        </tr>
        '''
    html_content += '</table>\n'


html_content += '''
</body>
</html>
'''


html_file_path = '/Users/moustafahegazy/Documents/GitHub/scheduler/grad_proj/output_data/schedule_output.html'

with open(html_file_path, 'w') as f:
    f.write(html_content)

print(f"Schedule has been generated and saved to '{html_file_path}'")


