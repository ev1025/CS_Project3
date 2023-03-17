import os
import csv
import sqlite3
from unicodedata import category

csv_path = os.path.join(os.getcwd(), 'metabase.csv')



conn = sqlite3.connect('metabase.db')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS LIST')
cur.execute('''CREATE TABLE LIST(
    name VARCHAR(50),
    score int,
    category VARCHAR(50)
)''')
with open(csv_path, 'r', encoding='UTF-8') as f:
    reader = csv.reader(f)
    for i in reader:
        cur.execute('''INSERT INTO LIST (name, score, category) VALUES (?,?,?)''',(i[1],i[2],i[3]))

conn.commit()