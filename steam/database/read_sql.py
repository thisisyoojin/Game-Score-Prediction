import mysql.connector as mysql
from database import config

def read_sql(fpath):
    """
    As the sql file is too big to read at once, I implement this function to read sql file.
    This function reads the line from top and executing one by one.

    INPUT:
    fpath(str) → the sql file location
    """
    
    # Creating connection
    conn = mysql.connect(**config)
    cursor = conn.cursor()
    # Set autocommit as default
    cursor.execute("SET autocommit = 1;")
    # Save a query
    lines = []
    
    # Read .sql file
    with open(fpath, "r") as f:
        
        while True:
            # Read the text line by line
            line = f.readline()

            if line is None:
                break
            
            if (line.startswith('--')) or (line.startswith('/*')):
                continue

            if len(line) > 1:
                lines.append(line)
                if ';' in line:
                    query = ''.join(lines).strip('\n')
                    cursor.execute(query)
                    lines = []
    
    conn.close()

