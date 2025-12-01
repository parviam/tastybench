from util import inference, extract_str
import ollama
from typing import List, Tuple, Dict
import requests, pathlib, re, json
import pandas as pd
from tqdm import tqdm
import asyncio

class PaperDataset():
    def __init__(self, name: str, paper_ids: List[str], model: str, client: ollama.Client|None=None, extract_prompt: str='model-elicitation/prompts/extract_idea.md', method: str|tuple=None):
        """
        Initialize a PaperDataset for extracting and storing paper ideas.

        Parameters
        ----------
        name : str
            Name of the dataset for identification purposes.
        paper_ids : List[str]
            List of Semantic Scholar paper IDs to process.
        model : str
            Model identifier for LLM inference.
        client : ollama.Client | None, optional
            Ollama client for local inference, if applicable (default is None).
        extract_prompt : str, optional
            Path to the prompt template file for idea extraction
            (default is 'model-elicitation/prompts/extract_idea.md').
        method : str | tuple, optional
            Method for obtaining paper content. Options:
            - None: Fetch abstracts and extract ideas (default)
            - 'tldr': Use TL;DR summaries as abstracts
            - ('intro_and_methods', json_path): Load intro/methods text from JSON file
        """
        self.model: str = model
        self.name: str = name
        self.client: ollama.Client | None = client
        self.paper_data: Dict[str, List[str]] = {
            'paper_id' : paper_ids,
        }
        self.method = method
        if not method:
            self.get_abstracts()
            self.extract_ideas(extract_prompt=extract_prompt)
        elif method == 'tldr':
            self.get_tldrs_as_abstracts()
        elif isinstance(method, tuple) and len(method) == 2 and method[0] == 'intro_and_methods':
            # method should be a tuple: ('intro_and_methods', json_path)
            self.get_intro_and_methods(method[1])
        else:
            raise ValueError("method must be None, 'tldr', or ('intro_and_methods', json_path)")

    def get_tldrs_as_abstracts(self) -> None:
        """
        Fetch TL;DR summaries from Semantic Scholar API and use them as abstracts.

        Updates the paper_data dictionary with paper_id, title, abstract (from TL;DR),
        and idea (same as TL;DR) for papers that have TL;DR summaries.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("TLDR fetching not implemented yet")
        r = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': ['title', 'tldr']},
            json={"ids": self.paper_data["paper_id"]}
        ).json()
        papers_with_tldrs_ids = []
        abstracts = []
        titles = []
        for paper in r:
            if 'tldr' in paper and paper['tldr'] is not None and paper['tldr']['text'] is not None:
                abstracts.append(paper['tldr']['text'].strip())
                titles.append(paper['title'].strip())
                papers_with_tldrs_ids.append(paper['paperId'])
       
        self.paper_data['paper_id'] = papers_with_tldrs_ids
        self.paper_data['abstract'] = abstracts
        self.paper_data['idea'] = abstracts  # Use tldr as idea directly
        self.paper_data['title'] = titles
    
    def get_intro_and_methods(self, json_path: str) -> None:
        """
        Load intro and methods text from a JSON file and use it as abstracts.

        The JSON file must contain a list of entries with the form:
            {
            "paperId": "...",
            "title": "...",
            "intro_and_methods": "<extracted text>",
            "success": true/false,
            "error": "<message if any>",
            "source_url": "<arxiv_html>"
            }

        Only entries with success == True and non-empty intro_and_methods
        are kept. paper_data['abstract'] is set to this text, and papers
        without valid intro_and_methods are removed.
        
        Parameters
        ----------
        json_path : str
            Path to the JSON file.
        """
        import json as json_module
        with open(json_path, "r") as f:
            data = json_module.load(f)

        # Create a mapping from paperId to entry
        data_map = {entry["paperId"]: entry for entry in data if "paperId" in entry}

        new_paper_ids: List[str] = []
        abstracts: List[str] = []
        titles: List[str] = []

        for pid in self.paper_data["paper_id"]:
            entry = data_map.get(pid)
            if (
                entry is None
                or not entry.get("success", False)
                or not entry.get("intro_and_methods")
            ):
                continue
            new_paper_ids.append(pid)
            abstracts.append(entry["intro_and_methods"].strip()[:30000]) # Truncate to 30,000 chars
            titles.append(entry.get("title", "").strip())

        self.paper_data["paper_id"] = new_paper_ids
        self.paper_data["abstract"] = abstracts
        self.paper_data["idea"] = abstracts  # Use full intro_and_methods as ideas directly
        self.paper_data["title"] = titles

    def get_abstracts(self) -> None:
        """
        Fetch paper abstracts and titles from Semantic Scholar API.

        Updates the paper_data dictionary with paper_id, abstract, and title
        for papers that have non-empty abstracts. Papers without abstracts are
        filtered out.
        """
        r = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': ['abstract', 'title']},
            json={"ids": self.paper_data["paper_id"]}
        ).json()
        papers_with_abstracts_ids = []
        abstracts = []
        titles = []
        for paper in r:
            if 'abstract' in paper and paper['abstract'].strip() != "":
                abstracts.append(paper['abstract'].strip())
                titles.append(paper['title'].strip())
                papers_with_abstracts_ids.append(paper['paperId'])
       
        self.paper_data['paper_id'] = papers_with_abstracts_ids
        self.paper_data['abstract'] = abstracts
        self.paper_data['title'] = titles
    
    def extract_ideas(self, extract_prompt: str) -> None:
        """
        Extract research ideas from paper abstracts using LLM inference.

        Uses the provided prompt template to extract concise ideas from each
        paper's abstract. The ideas are parsed from the model's response using
        XML-style tags (<idea>...</idea>).

        Parameters
        ----------
        extract_prompt : str
            Path to the prompt template file for idea extraction.
        """
        prompt_template = extract_str(extract_prompt)
        ideas = []
        for abstract in tqdm(self.paper_data['abstract'], desc="Extracting ideas"):
            prompt = prompt_template.replace('[ABSTRACT]', abstract)
            __, response = inference(prompt, model=self.model, client=self.client)
            idea = re.search(r'<idea>(.*?)</idea>', response, re.DOTALL)
            idea = idea.group(1).strip() if idea else "N/A"
            ideas.append(idea)
        self.paper_data['idea'] = ideas
    
    def export_to_csv(self, filename: str) -> None:
        """
        Export the paper dataset to a CSV file.

        Parameters
        ----------
        filename : str
            Path where the CSV file should be saved.
        """
        df = pd.DataFrame(self.paper_data)
        df.to_csv(filename, index=False)
    
    @classmethod
    def load_from_csv(cls, filename: str, model: str='openai/gpt-oss-120b', client: ollama.Client | None=None):
        """
        Load a PaperDataset from a previously saved CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file to load.
        model : str, optional
            Model identifier for future inference (default is 'openai/gpt-oss-120b').
        client : ollama.Client | None, optional
            Ollama client for local inference, if applicable (default is None).

        Returns
        -------
        PaperDataset
            A new PaperDataset instance loaded with data from the CSV file.
        """
        df = pd.read_csv(filename)
        dataset = cls(
            paper_ids = df['paper_id'].tolist(),
            model = model,
            client = client,
            name = filename.split('/')[-1].replace('.csv', '')
        )
        dataset.paper_data = {
            'paper_id': df['paper_id'].tolist(),
            'abstract': df['abstract'].tolist(),
            'title': df['title'].tolist(),
            'idea': df['idea'].tolist()
        }
        return dataset

class RankingDataset():
    def __init__(self, paper_dataset: PaperDataset, model: str, log: str, client: ollama.Client|None=None, epochs: int=1, 
                 ranking_prompt: str='model-elicitation/prompts/judge_ideas_goodhart.md', extract_choice_prompt: str='model-elicitation/prompts/extract_choice.md'):
        """
        Initialize a RankingDataset for pairwise comparison of paper ideas.

        Parameters
        ----------
        paper_dataset : PaperDataset
            Dataset containing papers and their extracted ideas to rank.
        model : str
            Model identifier for LLM inference used in judging comparisons.
        log : str
            Path to the log file for recording comparison details and errors.
        client : ollama.Client | None, optional
            Ollama client for local inference, if applicable (default is None).
        epochs : int, optional
            Number of epochs (complete passes through pairwise comparisons)
            to perform (default is 1).
        ranking_prompt : str, optional
            Path to the prompt template for judging idea comparisons
            (default is 'model-elicitation/prompts/judge_ideas_goodhart.md').
        extract_choice_prompt : str, optional
            Path to the prompt template for extracting the judge's choice
            (default is 'model-elicitation/prompts/extract_choice.md').
        """
        self.model: str = model
        self.client: ollama.Client | None = client
        self.paper_dataset: PaperDataset = paper_dataset
        self.ranking_data: pd.DataFrame = pd.DataFrame()
        self.epochs = epochs
        self.log = log
        self.judge_rankings(ranking_prompt, extract_choice_prompt)
    
    def append_to_log(self, message: str) -> None:
        """
        Append a message to the ranking log file.

        Parameters
        ----------
        message : str
            The message to append to the log file. The model name and separator
            lines are automatically added.
        """
        with open(self.log, 'a') as f:
            message += "="*80 + '\nmodel: ' + self.model + '\n'
            f.write(message + '\n' + "="*80 + '\n')
    
    def judge_rankings(self, ranking_prompt, extract_prompt) -> None:
        """
        Perform pairwise comparisons of paper ideas using LLM judgment.

        Conducts multiple epochs of pairwise comparisons between ideas, using
        the specified model to judge which idea is better. Handles errors
        gracefully and saves progress even if the process is interrupted.

        Parameters
        ----------
        ranking_prompt : str
            Path to the prompt template for judging idea comparisons.
        extract_prompt : str
            Path to the prompt template for extracting the choice from the
            judge's response.

        Raises
        ------
        Exception
            If inference fails during comparison, saves collected rankings and
            raises an exception with details about the failure point.
        """
        ranking_prompt_template = extract_str(ranking_prompt)
        extract_choice_template = extract_str(extract_prompt)
        rankings: List[Tuple[str, str]] = []
        df = pd.DataFrame(self.paper_dataset.paper_data)
        total_comparisons = self.epochs * (len(df['idea']) - 1)
        try:
            with tqdm(total=total_comparisons, desc=f"Getting {self.model} choices") as pbar:
                for epoch in range(self.epochs):
                    epoch_dataset = df.sample(frac=1).reset_index(drop=True)
                    for i in range(0, len(epoch_dataset['idea']) - 1):
                        idea1 = epoch_dataset['idea'][i]
                        idea2 = epoch_dataset['idea'][i+1]
                        prompt = ranking_prompt_template.replace('[PROJECT 1]', idea1).replace('[PROJECT 2]', idea2)
                        try:
                            __, response = inference(prompt, model=self.model, client=self.client)
                        except Exception as e:
                            error = f"Error at Epoch {epoch+1}, Comparison {i+1}:\n{str(e)}\nPrompt:\n{prompt}\n"
                            self.append_to_log(error)
                            print(f'[red]{error}')
                            self.ranking_data = pd.DataFrame(rankings, columns=['better_paper_id', 'worse_paper_id'])
                            raise Exception(f"Inference failed at epoch {epoch+1}, comparison {i+1}. Saved {len(rankings)} comparisons so far.") from e
                        
                        prompt = extract_choice_template.replace('[TRANSCRIPT]', response)
                        __, response = inference(prompt, model='openai/gpt-oss-120b', client=None)
                        if pbar.n % 50 == 0:
                            self.append_to_log(f"Epoch {epoch+1}, Comparison {i+1}:\nPrompt:\n{prompt}\nResponse:\n{response}\n")
                        if "UNCLEAR" in response:
                            self.append_to_log(f"Epoch {epoch+1}, Comparison {i+1} was unclear. Skipping.\nPrompt:\n{prompt}\nResponse:\n{response}\n")
                            pbar.update(1)
                            continue
                        elif "PROJECT 1" in response:
                            rankings.append((epoch_dataset['paper_id'][i], epoch_dataset['paper_id'][i+1]))
                        elif "PROJECT 2" in response:
                            rankings.append((epoch_dataset['paper_id'][i+1], epoch_dataset['paper_id'][i]))
                        pbar.update(1)
        finally:
            # Always save whatever rankings we collected
            self.ranking_data = pd.DataFrame(rankings, columns=['better_paper_id', 'worse_paper_id'])
            if len(rankings) > 0:
                self.append_to_log(f"Collected {len(rankings)} total comparisons before completion/failure.\n")
    
    def export_to_csv(self, filename: str) -> None:
        """
        Export the ranking data to a CSV file.

        Parameters
        ----------
        filename : str
            Path where the CSV file should be saved.
        """
        self.ranking_data.to_csv(filename, index=False)
    
    @classmethod
    def load_from_csv(cls, filename: str, paper_dataset: PaperDataset, model: str='openai/gpt-oss-120b', client: ollama.Client | None=None):
        """
        Load a RankingDataset from a previously saved CSV file.

        Parameters
        ----------
        filename : str
            Path to the CSV file containing ranking data.
        paper_dataset : PaperDataset
            The paper dataset associated with these rankings.
        model : str, optional
            Model identifier for future inference (default is 'openai/gpt-oss-120b').
        client : ollama.Client | None, optional
            Ollama client for local inference, if applicable (default is None).

        Returns
        -------
        RankingDataset
            A new RankingDataset instance loaded with data from the CSV file.
        """
        df = pd.read_csv(filename)
        ranking_dataset = cls(
            paper_dataset = paper_dataset,
            model = model,
            client = client,
            log = filename.replace('.csv', '.log'),
            epochs = 1  # Placeholder, actual epochs info not stored in CSV
        )
        ranking_dataset.ranking_data = df
        return ranking_dataset
    
def get_elo_rankings(ranking_dataset: RankingDataset, k_factor: float = 32.0, initial_rating: float = 1500.0) -> pd.DataFrame:
    """
    Calculate ELO rankings from pairwise comparisons.
    
    Args:
        ranking_dataset: RankingDataset containing pairwise comparison data
        k_factor: ELO K-factor (default 32.0)
        initial_rating: Initial ELO rating for all papers (default 1500.0)
    
    Returns:
        DataFrame with columns ['paper_id', 'elo_rating'] sorted by rating descending
    """
    # Initialize ratings dictionary
    elo_ratings = {}
    all_paper_ids = set(ranking_dataset.ranking_data['better_paper_id'].tolist() + 
                        ranking_dataset.ranking_data['worse_paper_id'].tolist())
    
    for paper_id in all_paper_ids:
        elo_ratings[paper_id] = initial_rating
    
    # Process each comparison
    for _, row in ranking_dataset.ranking_data.iterrows():
        winner_id = row['better_paper_id']
        loser_id = row['worse_paper_id']
        
        # Get current ratings
        winner_rating = elo_ratings[winner_id]
        loser_rating = elo_ratings[loser_id]
        
        # Calculate expected scores
        expected_winner = 1.0 / (1.0 + 10.0 ** ((loser_rating - winner_rating) / 400.0))
        expected_loser = 1.0 / (1.0 + 10.0 ** ((winner_rating - loser_rating) / 400.0))
        
        # Update ratings (winner gets score of 1, loser gets score of 0)
        elo_ratings[winner_id] = winner_rating + k_factor * (1.0 - expected_winner)
        elo_ratings[loser_id] = loser_rating + k_factor * (0.0 - expected_loser)
    
    # Convert to DataFrame and sort
    elo_df = pd.DataFrame([
        {'paper_id': paper_id, 'elo_rating': rating}
        for paper_id, rating in elo_ratings.items()
    ])
    elo_df = elo_df.sort_values('elo_rating', ascending=False).reset_index(drop=True)
    
    return elo_df

async def get_elo_rankings_for_model(model: str, paper_dataset: PaperDataset, output_dir: str, epochs: int=10, client: ollama.Client|None=None,
                                     ranking_prompt: str='model-elicitation', extract_choice_prompt: str='model-elicitation/prompts/extract_choice.md') -> None:
    """
    Asynchronously generate ELO rankings for a paper dataset using a specified model.

    Runs pairwise comparisons and computes ELO rankings in a separate thread to
    avoid blocking the event loop. Saves both the raw ranking comparisons and
    computed ELO scores to CSV files.

    Parameters
    ----------
    model : str
        Model identifier for LLM inference used in judging comparisons.
    paper_dataset : PaperDataset
        Dataset containing papers and their extracted ideas to rank.
    output_dir : str
        Directory path where output files (rankings.csv, elo.csv, ranking.log)
        will be saved.
    epochs : int, optional
        Number of epochs for pairwise comparisons (default is 10).
    client : ollama.Client | None, optional
        Ollama client for local inference, if applicable (default is None).
    ranking_prompt : str, optional
        Path to the prompt template for judging comparisons
        (default is 'model-elicitation').
    extract_choice_prompt : str, optional
        Path to the prompt template for extracting choices
        (default is 'model-elicitation/prompts/extract_choice.md').
    """
    # Run the blocking operations in a separate thread
    def _run_blocking():
        log_file = output_dir + 'ranking.log'
        ranking_dataset = RankingDataset(paper_dataset, model=model, epochs=epochs, client=client, 
                                         ranking_prompt=ranking_prompt, extract_choice_prompt=extract_choice_prompt, log=log_file)
        ranking_dataset.export_to_csv(output_dir + 'rankings.csv')
        return ranking_dataset
    
    ranking_dataset = await asyncio.to_thread(_run_blocking)
    await asyncio.to_thread(lambda: get_elo_rankings(ranking_dataset=ranking_dataset).to_csv(output_dir + 'elo.csv', index=False))

class Experiment:
    def __init__(self, name: str, paper_dataset: PaperDataset, models: List[Tuple[str, str]], epochs: List[int]=[10], client: ollama.Client|None=None,
                 ranking_prompt: str='model-elicitation/prompts/judge_ideas_goodhart.md', extract_choice_prompt: str='model-elicitation/prompts/extract_choice.md'):
        """
        Initialize an experiment for running ELO ranking across multiple models and epochs.

        Sets up the directory structure and saves metadata for tracking experimental
        configurations.

        Parameters
        ----------
        name : str
            Name of the experiment, used for output directory naming.
        paper_dataset : PaperDataset
            Dataset containing papers to rank.
        models : List[Tuple[str, str]]
            List of (model_id, model_name) tuples where model_id is the identifier
            for inference and model_name is used for directory naming.
        epochs : List[int], optional
            List of epoch counts to test (default is [10]).
        client : ollama.Client | None, optional
            Ollama client for local inference, if applicable (default is None).
        ranking_prompt : str, optional
            Path to the prompt template for judging comparisons
            (default is 'model-elicitation/prompts/judge_ideas_goodhart.md').
        extract_choice_prompt : str, optional
            Path to the prompt template for extracting choices
            (default is 'model-elicitation/prompts/extract_choice.md').
        """
        self.name = name
        self.paper_dataset = paper_dataset
        self.models = models
        self.epochs = epochs
        self.client = client
        self.ranking_prompt = ranking_prompt
        self.extract_choice_prompt = extract_choice_prompt
        self.output_dir = f"model-elicitation/data/{name}/{models}/"
        pathlib.Path(f"model-elicitation/data/{self.name}/").mkdir(parents=True, exist_ok=True)
        for model, model_name in models:
            pathlib.Path(f"model-elicitation/data/{self.name}/{model_name}/").mkdir(parents=True, exist_ok=True)
            for epoch in epochs:
                pathlib.Path(f"model-elicitation/data/{self.name}/{model_name}/{epoch}-epochs/").mkdir(parents=True, exist_ok=True)
        self.save_metadata()
    
    def save_metadata(self) -> None:
        """
        Save experiment configuration metadata to a JSON file.

        Stores information about the experiment including name, models, epochs,
        paper dataset, and prompt paths for reproducibility.
        """
        metadata = {
            'name': self.name,
            'models': [model for model, _ in self.models],
            'epochs': self.epochs,
            'paper_dataset': self.paper_dataset.name, 
            'ranking_prompt': self.ranking_prompt,
            'extract_choice_prompt': self.extract_choice_prompt
        }
        with open(f"model-elicitation/data/{self.name}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    async def run(self, tg: asyncio.TaskGroup) -> None:
        """
        Execute the experiment by creating ranking tasks for all model/epoch combinations.

        Creates asynchronous tasks within the provided TaskGroup for each combination
        of model and epoch count specified in the experiment configuration.

        Parameters
        ----------
        tg : asyncio.TaskGroup
            The TaskGroup to which ranking tasks will be added for concurrent execution.
        """
        for model, model_name in self.models:
            for epoch in self.epochs:
                output_dir = f"model-elicitation/data/{self.name}/{model_name}/{epoch}-epochs/"
                print("Created task: ", model, epoch, self.name)
                tg.create_task(get_elo_rankings_for_model(
                    model=model,
                    paper_dataset=self.paper_dataset,
                    output_dir=output_dir,
                    epochs=epoch,
                    client=self.client,
                    ranking_prompt=self.ranking_prompt,
                    extract_choice_prompt=self.extract_choice_prompt
                ))

async def main():
    rl_csv = 'model-elicitation/data/llm_rl_yix_curate.csv'
    paper_ids = pd.read_csv(rl_csv)
    paper_ids = paper_ids['paperId'].to_list()

    llm_rl_curated = PaperDataset(paper_ids=paper_ids, name='llm-rl-yix-curate', model='openai/gpt-oss-120b', extract_prompt='model-elicitation/prompts/curated/extract_idea.md', method=None)
    llm_rl_curated.export_to_csv('model-elicitation/data/llm_rl_yix_curate_with_ideas.csv')

    models = [
        ('claude-sonnet-4-5-20250929', 'claude-sonnet-4-5'),
        ('gpt-5.1', 'gpt-5-1'),
        ('gemini-2.5-pro', 'gemini-2-5-pro'),
    ]
    epochs = [20]

    max_goodhart_experiment = Experiment(
        name='max-goodhart-curated',
        paper_dataset=llm_rl_curated,
        models=models,
        epochs=epochs,
        ranking_prompt='model-elicitation/prompts/curated/judge_ideas_goodhart_curated_max.md'
    )

    goodhart_experiment = Experiment(
        name='goodhart-curated',
        paper_dataset=llm_rl_curated,
        models=models,
        epochs=epochs,
        ranking_prompt='model-elicitation/prompts/curated/judge_ideas_goodhart_curated.md'
    )

    curated_experiment = Experiment(
        name='curated',
        paper_dataset=llm_rl_curated,
        models=models,
        epochs=epochs,
        ranking_prompt='model-elicitation/prompts/curated/judge_ideas_curated.md'
    )
    
    experiments = [goodhart_experiment, curated_experiment, max_goodhart_experiment]

    async with asyncio.TaskGroup() as tg:
        for experiment in experiments:
            tg.create_task(experiment.run(tg)) 


async def intro_and_methods_exp():
    rl_csv = 'model-elicitation/data/llm_rl_yix_curate.csv'
    paper_ids = pd.read_csv(rl_csv)
    paper_ids = paper_ids['paperId'].to_list()

    llm_rl_intro_methods = PaperDataset(
        paper_ids=paper_ids,
        name='llm-rl-yix-curate-intro-methods',
        model='openai/gpt-oss-120b',
        extract_prompt='model-elicitation/prompts/curated/extract_idea.md',
        method=('intro_and_methods', 'model-elicitation/data/llm_rl_yix_curate_intro_methods.json')
    )
    llm_rl_intro_methods.export_to_csv('model-elicitation/data/llm_rl_yix_curate_intro_methods_with_ideas.csv')

    models = [
        ('claude-sonnet-4-5-20250929', 'claude-sonnet-4-5'),
        ('gpt-5.1', 'gpt-5-1'),
        ('gemini-2.5-pro', 'gemini-2-5-pro'),
    ]
    epochs = [50]

    intro_methods_experiment = Experiment(
        name='intro-methods-curated',
        paper_dataset=llm_rl_intro_methods,
        models=models,
        epochs=epochs,
        ranking_prompt='model-elicitation/prompts/judge_intro_methods.md'
    )

    async with asyncio.TaskGroup() as tg:
        await intro_methods_experiment.run(tg)

if __name__ == "__main__":
    asyncio.run(intro_and_methods_exp())