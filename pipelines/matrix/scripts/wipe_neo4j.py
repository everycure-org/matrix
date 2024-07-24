"""Wipes neo4j instance locally."""
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os


# checking if any existing env variables set the n4j password which hints at prod port forwarding
def avoid_wiping_prod():
    if os.environ.get("NEO4J_PASSWORD", "") != "":
        print(
            "careful, you may be accidentally wiping a remote neo, saving you from this now!"
        )
        exit()


def connect_to_neo4j():
    return GraphDatabase.driver(
        os.environ["NEO4J_HOST"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def get_user_databases(session):
    result = session.run("SHOW DATABASES")
    return [record["name"] for record in result if record["name"] != "system"]


def main():
    try:
        with connect_to_neo4j() as driver:
            with driver.session(database="system") as system_session:
                databases = get_user_databases(system_session)
                print(f"Found {len(databases)} user database(s)")
                print(databases)

                for db in databases:
                    print(f"dropping {db}")
                    system_session.run(f"DROP DATABASE {db} IF EXISTS")

            print("\nAll user databases have been wiped.")

    except Exception as ex:
        print(f"An error occurred: {ex}")


if __name__ == "__main__":
    avoid_wiping_prod()
    load_dotenv()
    main()
