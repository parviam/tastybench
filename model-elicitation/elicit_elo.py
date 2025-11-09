from util import inference, extract_str
import ollama
from typing import List, Tuple, Dict
import requests, pathlib, re, json
import pandas as pd
from tqdm import tqdm
import asyncio

class PaperDataset():
    def __init__(self, name: str, paper_ids: List[str], model: str, client: ollama.Client|None=None, extract_prompt: str='model-elicitation/prompts/extract_idea.md', tldr: bool=False):
        self.model: str = model
        self.name: str = name
        self.client: ollama.Client | None = client
        self.paper_data: Dict[str, List[str]] = {
            'paper_id' : paper_ids,
        }
        self.tldr = tldr
        if not tldr:
            self.get_abstracts()
            self.extract_ideas(extract_prompt=extract_prompt)
        else:
            self.get_tldrs_as_abstracts()

    def get_tldrs_as_abstracts(self) -> None:
        r = requests.post(
            'https://api.semanticscholar.org/graph/v1/paper/batch',
            params={'fields': ['tldr', 'title']},
            json={"ids": self.paper_data["paper_id"]}
        ).json()
        papers_with_tldrs_ids = []
        abstracts = []
        titles = []
        for paper in r:
            if 'tldr' in paper and paper['tldr'] is not None and paper['tldr']['text'] is not None and paper['tldr']['text'].strip() != "":
                abstracts.append(paper['tldr']['text'].strip())
                titles.append(paper['title'].strip())
                papers_with_tldrs_ids.append(paper['paperId'])
       
        self.paper_data['paper_id'] = papers_with_tldrs_ids
        self.paper_data['abstract'] = abstracts
        self.paper_data['idea'] = abstracts  # Use tldr as idea directly
        self.paper_data['title'] = titles
    
    def get_abstracts(self) -> None:
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
        df = pd.DataFrame(self.paper_data)
        df.to_csv(filename, index=False)
    
    @classmethod
    def load_from_csv(cls, filename: str, model: str='openai/gpt-oss-120b', client: ollama.Client | None=None):
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
    def __init__(self, paper_dataset: PaperDataset, model: str, client: ollama.Client|None=None, epochs: int=1, 
                 ranking_prompt: str='model-elicitation/prompts/judge_ideas_goodhart.md', extract_prompt: str='model-elicitation/prompts/extract_choice.md'):
        self.model: str = model
        self.client: ollama.Client | None = client
        self.paper_dataset: PaperDataset = paper_dataset
        self.ranking_data: pd.DataFrame = pd.DataFrame()
        self.epochs = epochs
        self.judge_rankings(ranking_prompt, extract_prompt)
    
    def judge_rankings(self, ranking_prompt, extract_prompt) -> None:
        ranking_prompt_template = extract_str(ranking_prompt)
        extract_choice_template = extract_str(extract_prompt)
        rankings: List[Tuple[str, str]] = []
        df = pd.DataFrame(self.paper_dataset.paper_data)
        total_comparisons = self.epochs * (len(df['idea']) - 1)
        with tqdm(total=total_comparisons, desc=f"Getting {self.model} choices") as pbar:
            for epoch in range(self.epochs):
                epoch_dataset = df.sample(frac=1).reset_index(drop=True)
                for i in range(0, len(epoch_dataset['idea']) - 1):
                    idea1 = epoch_dataset['idea'][i]
                    idea2 = epoch_dataset['idea'][i+1]
                    prompt = ranking_prompt_template.replace('[PROJECT 1]', idea1).replace('[PROJECT 2]', idea2)
                    __, response = inference(prompt, model=self.model, client=self.client)

                    prompt = extract_choice_template.replace('[TRANSCRIPT]', response)
                    __, response = inference(prompt, model=self.model, client=self.client)
                    if "UNCLEAR" in response:
                        print(prompt)
                        pbar.update(1)
                        continue
                    elif "PROJECT 1" in response:
                        rankings.append((epoch_dataset['paper_id'][i], epoch_dataset['paper_id'][i+1]))
                    elif "PROJECT 2" in response:
                        rankings.append((epoch_dataset['paper_id'][i+1], epoch_dataset['paper_id'][i]))
                    pbar.update(1)
        self.ranking_data = pd.DataFrame(rankings, columns=['better_paper_id', 'worse_paper_id'])
    
    def export_to_csv(self, filename: str) -> None:
        self.ranking_data.to_csv(filename, index=False)
    
    @classmethod
    def load_from_csv(cls, filename: str, paper_dataset: PaperDataset, model: str='openai/gpt-oss-120b', client: ollama.Client | None=None):
        df = pd.read_csv(filename)
        ranking_dataset = cls(
            paper_dataset = paper_dataset,
            model = model,
            client = client
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
    # Run the blocking operations in a separate thread
    def _run_blocking():
        ranking_dataset = RankingDataset(paper_dataset, model=model, epochs=epochs, client=client, 
                                         ranking_prompt=ranking_prompt, extract_prompt=extract_choice_prompt)
        ranking_dataset.export_to_csv(output_dir + 'rankings.csv')
        return ranking_dataset
    
    ranking_dataset = await asyncio.to_thread(_run_blocking)
    await asyncio.to_thread(lambda: get_elo_rankings(ranking_dataset=ranking_dataset).to_csv(output_dir + 'elo.csv', index=False))

class Experiment:
    def __init__(self, name: str, paper_dataset: PaperDataset, models: List[Tuple[str, str]], epochs: List[int]=[10], client: ollama.Client|None=None,
                 ranking_prompt: str='model-elicitation/prompts/judge_ideas_goodhart.md', extract_choice_prompt: str='model-elicitation/prompts/extract_choice.md'):
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
        metadata = {
            'name': self.name,
            'models': [model for model, _ in self.models],
            'epochs': self.epochs,
            'paper_ids': self.paper_dataset.paper_data['paper_id'],
            'tldr_used': self.paper_dataset.tldr,

        }
        with open(f"model-elicitation/data/{self.name}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
    
    async def run(self, tg: asyncio.TaskGroup) -> None:
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
    rl_csv = 'model-elicitation/data/llm_rl.csv'
    paper_ids = pd.read_csv(rl_csv)
    paper_ids = paper_ids['paperId'].to_list()
    tldr_dataset = PaperDataset(paper_ids, model='openai/gpt-oss-120b', tldr=True)
    tldr_dataset.export_to_csv('model-elicitation/data/llm_rl_with_tldrs.csv')
    max_goodhart_dataset = PaperDataset(paper_ids, model='openai/gpt-oss-120b', extract_prompt='model-elicitation/prompts/judge_ideas_goodhart_max.md')
    max_goodhart_dataset.export_to_csv('model-elicitation/data/llm_rl_with_max_goodhart_ideas.csv')

    models = [
        ('openai/gpt-oss-120b', 'gpt-oss-120b'),
        ('openai/gpt-oss-20b', 'gpt-oss-20b'),
        ('meta-llama/Llama-3.3-70B-Instruct', 'llama-3-70b'),
    ]
    experiments = ['tldr-10-epochs', 'tldr-30-epochs', 'max-goodhart-10-epochs', 'max-goodhart-30-epochs']
    for exp in experiments:
        pathlib.Path(f"model-elicitation/data/{exp}/").mkdir(parents=True, exist_ok=True)
        for model, name in models:
            pathlib.Path(f"model-elicitation/data/{exp}/{name}/").mkdir(parents=True, exist_ok=True)

    async with asyncio.TaskGroup() as tg:
        for exp in experiments:
            for model, name in models:
                output_dir = f"model-elicitation/data/{exp}/{name}/"
                epochs = 10 if '10-epochs' in exp else 30
                tg.create_task(get_elo_rankings_for_model(
                    model=model,
                    paper_dataset=dataset,
                    output_dir=output_dir,
                    epochs=epochs
                ))

if __name__ == "__main__":
    rl_csv = 'model-elicitation/data/llm_rl.csv'
    paper_ids = pd.read_csv(rl_csv)
    paper_ids = paper_ids['paperId'].to_list()
    dataset = PaperDataset(paper_ids, model='openai/gpt-oss-120b', tdlr=True)
    dataset.export_to_csv('model-elicitation/data/llm_rl_with_ideas.csv')
    #asyncio.run(main())