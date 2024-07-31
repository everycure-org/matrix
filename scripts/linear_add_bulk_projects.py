import typer
import csv
import openai
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from typing import Optional

app = typer.Typer()

# GraphQL query to fetch teams
FETCH_TEAMS_QUERY = gql("""
query {
  teams {
    nodes {
      id
      name
    }
  }
}
""")

# GraphQL mutation for creating a project
CREATE_PROJECT_MUTATION = gql("""
mutation CreateProject($input: ProjectCreateInput!) {
  projectCreate(input: $input) {
    success
    project {
      id
      name
      startDate
      targetDate
    }
  }
}
""")

# Add this GraphQL query to fetch projects for a team
FETCH_TEAM_PROJECTS_QUERY = gql("""
query FetchTeamProjects($teamId: String!) {
  team(id: $teamId) {
    projects {
      nodes {
        id
        name
      }
    }
  }
}
""")

# Add this GraphQL mutation to delete a project
DELETE_PROJECT_MUTATION = gql("""
mutation DeleteProject($id: String!) {
  projectDelete(id: $id) {
    success
  }
}
""")

def get_graphql_client(api_key):
    transport = RequestsHTTPTransport(
        url='https://api.linear.app/graphql',
        headers={'Authorization': api_key}
    )
    return Client(transport=transport, fetch_schema_from_transport=True)

def fetch_teams(client):
    result = client.execute(FETCH_TEAMS_QUERY)
    return {team['name']: team['id'] for team in result['teams']['nodes']}

def simplify_name(name, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that simplifies project names."},
            {"role": "user", "content": f"Simplify this project name, keeping the 'Subtask X.X' prefix and maintaining the core meaning, but make it shorter and crisper. By no means should the name be longer than 70 characters: '{name}'"}
        ]
    )
    return response.choices[0].message.content.strip()

@app.command()
def add(
    csv_file: str,
    team_name: str,
    linear_api_key: str = typer.Option(..., prompt=True, help="Your Linear API key"),
    openai_api_key: str = typer.Option(..., prompt=True, help="Your OpenAI API key")
):
    """Add projects from a CSV file to a Linear team."""
    client = get_graphql_client(linear_api_key)
    
    teams = fetch_teams(client)
    if team_name not in teams:
        print(f"Error: Team '{team_name}' not found. Available teams: {', '.join(teams.keys())}")
        return
    team_id = teams[team_name]
    
    with open(csv_file, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            original_name = row['Name']
            simplified_name = simplify_name(original_name, openai_api_key)
            
            variables = {
                "input": {
                    "name": simplified_name[0:79],
                    "startDate": row['Start Date'],
                    "targetDate": row['Due Date'],
                    "teamIds": [team_id],
                }
            }
            
            try:
                result = client.execute(CREATE_PROJECT_MUTATION, variable_values=variables)
                print(f"Created project: {result['projectCreate']['project']['name']}")
                print(f"Original name: {original_name}")
                print(f"Simplified name: {simplified_name}")
                print("---")
            except Exception as e:
                print(f"Error creating project {simplified_name}: {str(e)}")

@app.command()
def delete(
    team_name: str,
    linear_api_key: Optional[str] = typer.Option(None, envvar="LINEAR_API_KEY", prompt=True),
    confirm: Optional[bool] = typer.Option(None, "--confirm", prompt="Are you sure you want to delete all projects? This action cannot be undone.")
):
    if not confirm:
        print("Operation cancelled.")
        return

    client = get_graphql_client(linear_api_key)
    
    teams = fetch_teams(client)
    if team_name not in teams:
        print(f"Error: Team '{team_name}' not found. Available teams: {', '.join(teams.keys())}")
        return
    team_id = teams[team_name]
    
    variables = {"teamId": team_id}
    result = client.execute(FETCH_TEAM_PROJECTS_QUERY, variable_values=variables)
    projects = result['team']['projects']['nodes']
    
    if not projects:
        print(f"No projects found for team '{team_name}'.")
        return
    
    print(f"Found {len(projects)} projects. Deleting...")
    
    for project in projects:
        try:
            delete_variables = {"id": project['id']}
            delete_result = client.execute(DELETE_PROJECT_MUTATION, variable_values=delete_variables)
            if delete_result['projectDelete']['success']:
                print(f"Deleted project: {project['name']}")
            else:
                print(f"Failed to delete project: {project['name']}")
        except Exception as e:
            print(f"Error deleting project {project['name']}: {str(e)}")
    
    print("Project deletion operation completed.")

if __name__ == "__main__":
    app()