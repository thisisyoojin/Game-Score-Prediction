import pandas as pd
import mysql.connector as mysql
from database import config

queries = {
    "app_query": "SELECT `appid`, `Type`, `Price`, `Release_Date`, `Rating`, `Required_Age`, `Is_Multiplayer` FROM App_ID_Info_Old",
    "achievement_query": "SELECT appid, AVG(Percentage) as Achievement_rate FROM Achievement_Percentages GROUP BY appid",
    "developer_query": "SELECT * FROM Games_Developers_Old",
    "genre_query": "SELECT * FROM Games_Genres_Old",
    "publisher_query": "SELECT * FROM Games_Publishers_Old",
    "playing_query": """
                    SELECT appid, AVG(playtime_forever) as Playtime, COUNT(DISTINCT steamid) as Total_buyers FROM (
                        SELECT steamid, appid, playtime_forever, MAX(dateretrieved)
                        FROM Games_Daily
                        GROUP BY steamid, appid, playtime_forever
                        ) as latest
                    GROUP BY appid;
                    """
}


def read_df_from_query(query, conn):
    
    df = pd.read_sql(query, conn)
    df.set_index('appid', inplace=True)
    
    return df


def create_df():

    conn = mysql.connect(**config)
    app_df = read_df_from_query(queries['app_query'], conn)
    achievement_df = read_df_from_query(queries['achievement_query'], conn)
    developer_df = read_df_from_query(queries['developer_query'], conn)
    genre_df = read_df_from_query(queries['genre_query'], conn)
    publisher_df = read_df_from_query(queries['publisher_query'], conn)
    playing_df = read_df_from_query(queries['playing_query'], conn)
    conn.close()

    app_df = app_df.join(developer_df).join(genre_df).join(publisher_df).join(achievement_df).join(playing_df)
    app_df.dropna(subset=['Total_buyers'], inplace=True)
    X, y = app_df.iloc[:,:-1], app_df['Total_buyers']

    return X, y
