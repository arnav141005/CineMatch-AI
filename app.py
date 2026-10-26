
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_data = [
    ["Inception", "Movie", "Sci-Fi, Thriller", 2010, "A thief enters dreams to steal secrets and plant ideas.", "Leonardo DiCaprio, Joseph Gordon-Levitt", "Christopher Nolan"],
    ["Interstellar", "Movie", "Sci-Fi, Drama", 2014, "Astronauts travel through a wormhole to save humanity.", "Matthew McConaughey, Anne Hathaway", "Christopher Nolan"],
    ["The Dark Knight", "Movie", "Action, Crime, Drama", 2008, "Batman faces the Joker in a battle for Gotham City.", "Christian Bale, Heath Ledger", "Christopher Nolan"],
    ["Tenet", "Movie", "Action, Sci-Fi, Thriller", 2020, "A secret agent manipulates time to prevent a global disaster.", "John David Washington, Robert Pattinson", "Christopher Nolan"],
    ["The Prestige", "Movie", "Drama, Mystery, Thriller", 2006, "Two rival magicians compete to create the ultimate illusion.", "Hugh Jackman, Christian Bale", "Christopher Nolan"],
    ["Avatar", "Movie", "Sci-Fi, Adventure, Fantasy", 2009, "A marine explores an alien world filled with extraordinary life.", "Sam Worthington, Zoe Saldana", "James Cameron"],
    ["Titanic", "Movie", "Romance, Drama", 1997, "Two young people fall in love aboard a doomed ocean liner.", "Leonardo DiCaprio, Kate Winslet", "James Cameron"],
    ["The Matrix", "Movie", "Sci-Fi, Action", 1999, "A hacker discovers reality is a computer-generated simulation.", "Keanu Reeves, Laurence Fishburne", "Lana and Lilly Wachowski"],
    ["The Martian", "Movie", "Sci-Fi, Adventure, Drama", 2015, "An astronaut struggles to survive alone on Mars.", "Matt Damon, Jessica Chastain", "Ridley Scott"],
    ["Gravity", "Movie", "Sci-Fi, Thriller, Drama", 2013, "Astronauts fight to survive after an accident in space.", "Sandra Bullock, George Clooney", "Alfonso Cuaron"],
    ["Jurassic Park", "Movie", "Adventure, Sci-Fi, Thriller", 1993, "Scientists create a dinosaur theme park that becomes dangerous.", "Sam Neill, Laura Dern", "Steven Spielberg"],
    ["Avengers Endgame", "Movie", "Action, Sci-Fi, Adventure", 2019, "Superheroes unite to reverse a devastating cosmic catastrophe.", "Robert Downey Jr, Chris Evans", "Russo Brothers"],
    ["Iron Man", "Movie", "Action, Sci-Fi, Adventure", 2008, "A genius inventor builds an armored suit to fight evil.", "Robert Downey Jr, Gwyneth Paltrow", "Jon Favreau"],
    ["Doctor Strange", "Movie", "Fantasy, Action, Adventure", 2016, "A surgeon learns mystical arts to protect reality.", "Benedict Cumberbatch, Tilda Swinton", "Scott Derrickson"],
    ["Black Panther", "Movie", "Action, Adventure, Sci-Fi", 2018, "A king protects his technologically advanced African nation.", "Chadwick Boseman, Lupita Nyongo", "Ryan Coogler"],
    ["Guardians of the Galaxy", "Movie", "Sci-Fi, Comedy, Adventure", 2014, "A group of space outlaws unite to protect the galaxy.", "Chris Pratt, Zoe Saldana", "James Gunn"],
    ["Spider-Man Homecoming", "Movie", "Action, Comedy, Adventure", 2017, "A young superhero balances school life and fighting criminals.", "Tom Holland, Michael Keaton", "Jon Watts"],
    ["The Shawshank Redemption", "Movie", "Drama, Crime", 1994, "A banker builds hope and friendship inside a prison.", "Tim Robbins, Morgan Freeman", "Frank Darabont"],
    ["The Godfather", "Movie", "Crime, Drama", 1972, "The aging leader of a crime family passes power to his son.", "Marlon Brando, Al Pacino", "Francis Ford Coppola"],
    ["Goodfellas", "Movie", "Crime, Drama, Biography", 1990, "A man rises through the ranks of an organized crime group.", "Ray Liotta, Robert De Niro", "Martin Scorsese"],
    ["Pulp Fiction", "Movie", "Crime, Thriller, Drama", 1994, "Interconnected crime stories unfold in unexpected ways.", "John Travolta, Samuel L Jackson", "Quentin Tarantino"],
    ["Fight Club", "Movie", "Drama, Thriller", 1999, "An office worker forms an underground fight club.", "Brad Pitt, Edward Norton", "David Fincher"],
    ["Se7en", "Movie", "Crime, Mystery, Thriller", 1995, "Detectives hunt a killer whose crimes follow seven deadly sins.", "Brad Pitt, Morgan Freeman", "David Fincher"],
    ["Shutter Island", "Movie", "Mystery, Thriller, Drama", 2010, "A detective investigates a disappearance at an isolated hospital.", "Leonardo DiCaprio, Mark Ruffalo", "Martin Scorsese"],
    ["The Social Network", "Movie", "Drama, Biography", 2010, "A college student creates a social networking company.", "Jesse Eisenberg, Andrew Garfield", "David Fincher"],
    ["Forrest Gump", "Movie", "Drama, Romance, Comedy", 1994, "A kindhearted man witnesses major events in American history.", "Tom Hanks, Robin Wright", "Robert Zemeckis"],
    ["The Pursuit of Happyness", "Movie", "Drama, Biography", 2006, "A struggling father works toward a better future for his son.", "Will Smith, Jaden Smith", "Gabriele Muccino"],
    ["La La Land", "Movie", "Romance, Drama, Music", 2016, "A musician and an actor pursue dreams and love in Los Angeles.", "Ryan Gosling, Emma Stone", "Damien Chazelle"],
    ["The Notebook", "Movie", "Romance, Drama", 2004, "A lifelong romance unfolds through memories and personal challenges.", "Ryan Gosling, Rachel McAdams", "Nick Cassavetes"],
    ["Joker", "Movie", "Crime, Drama, Thriller", 2019, "A troubled comedian descends into chaos in Gotham City.", "Joaquin Phoenix, Robert De Niro", "Todd Phillips"],
    ["Dune", "Movie", "Sci-Fi, Adventure, Drama", 2021, "A young noble travels to a desert planet rich in a valuable resource.", "Timothee Chalamet, Zendaya", "Denis Villeneuve"],
    ["Blade Runner 2049", "Movie", "Sci-Fi, Mystery, Drama", 2017, "A detective uncovers secrets about artificial humans and society.", "Ryan Gosling, Harrison Ford", "Denis Villeneuve"],
    ["Mad Max Fury Road", "Movie", "Action, Adventure, Sci-Fi", 2015, "Survivors escape across a dangerous post-apocalyptic desert.", "Tom Hardy, Charlize Theron", "George Miller"],
    ["John Wick", "Movie", "Action, Thriller, Crime", 2014, "A retired assassin returns to the criminal underworld.", "Keanu Reeves, Ian McShane", "Chad Stahelski"],
    ["Mission Impossible Fallout", "Movie", "Action, Thriller, Adventure", 2018, "A secret agent races to stop a dangerous international threat.", "Tom Cruise, Rebecca Ferguson", "Christopher McQuarrie"],
    ["The Conjuring", "Movie", "Horror, Mystery, Thriller", 2013, "Paranormal investigators help a family facing a terrifying haunting.", "Vera Farmiga, Patrick Wilson", "James Wan"],
    ["Get Out", "Movie", "Horror, Mystery, Thriller", 2017, "A man uncovers disturbing secrets during a visit to his partner's family.", "Daniel Kaluuya, Allison Williams", "Jordan Peele"],
    ["A Quiet Place", "Movie", "Horror, Thriller, Sci-Fi", 2018, "A family survives in silence while hunted by sound-sensitive creatures.", "Emily Blunt, John Krasinski", "John Krasinski"],
    ["Toy Story", "Movie", "Animation, Comedy, Adventure", 1995, "Toys secretly come alive and embark on imaginative adventures.", "Tom Hanks, Tim Allen", "John Lasseter"],
    ["Finding Nemo", "Movie", "Animation, Adventure, Comedy", 2003, "A clownfish crosses the ocean to find his missing son.", "Albert Brooks, Ellen DeGeneres", "Andrew Stanton"],
    ["Coco", "Movie", "Animation, Fantasy, Music", 2017, "A young musician visits the colorful world of the dead.", "Anthony Gonzalez, Gael Garcia Bernal", "Lee Unkrich"],
    ["Inside Out", "Movie", "Animation, Comedy, Drama", 2015, "Emotions guide a young girl's life during a major family move.", "Amy Poehler, Phyllis Smith", "Pete Docter"],
    ["The Lion King", "Movie", "Animation, Adventure, Drama", 1994, "A young lion learns to accept his responsibility as king.", "Matthew Broderick, James Earl Jones", "Roger Allers"],
    ["Stranger Things", "Web Series", "Sci-Fi, Horror, Mystery", 2016, "Children uncover supernatural mysteries in a small town.", "Millie Bobby Brown, Finn Wolfhard", "The Duffer Brothers"],
    ["Breaking Bad", "Web Series", "Crime, Drama, Thriller", 2008, "A chemistry teacher turns to producing illegal drugs after a diagnosis.", "Bryan Cranston, Aaron Paul", "Vince Gilligan"],
    ["Sherlock", "Web Series", "Crime, Mystery, Drama", 2010, "A brilliant detective solves complex crimes in modern London.", "Benedict Cumberbatch, Martin Freeman", "Mark Gatiss"],
    ["The Office", "Web Series", "Comedy", 2005, "Employees navigate awkward humor and everyday life at a paper company.", "Steve Carell, John Krasinski", "Greg Daniels"],
    ["Friends", "Web Series", "Comedy, Romance", 1994, "Six friends navigate relationships and careers in New York City.", "Jennifer Aniston, Courteney Cox", "David Crane"],
    ["Dark", "Web Series", "Sci-Fi, Mystery, Thriller", 2017, "A missing child leads four families into a time-travel mystery.", "Louis Hofmann, Lisa Vicari", "Baran bo Odar"],
    ["The Crown", "Web Series", "Drama, History, Biography", 2016, "A royal family faces political challenges and personal struggles.", "Claire Foy, Olivia Colman", "Peter Morgan"],
    ["The Mandalorian", "Web Series", "Sci-Fi, Action, Adventure", 2019, "A lone bounty hunter travels through a galaxy far from authority.", "Pedro Pascal, Giancarlo Esposito", "Jon Favreau"],
    ["Wednesday", "Web Series", "Fantasy, Mystery, Comedy", 2022, "A clever student investigates mysteries at a supernatural academy.", "Jenna Ortega, Emma Myers", "Alfred Gough"],
    ["Money Heist", "Web Series", "Crime, Thriller, Drama", 2017, "A criminal mastermind organizes a daring plan with a skilled crew.", "Ursula Corbero, Alvaro Morte", "Alex Pina"],
    ["The Witcher", "Web Series", "Fantasy, Action, Adventure", 2019, "A monster hunter navigates a dangerous world of magic and destiny.", "Henry Cavill, Anya Chalotra", "Lauren Schmidt Hissrich"],
    ["Squid Game", "Web Series", "Thriller, Drama, Mystery", 2021, "Heavily indebted contestants enter dangerous games for a prize.", "Lee Jung-jae, Jung Ho-yeon", "Hwang Dong-hyuk"],
    ["The Queen's Gambit", "Web Series", "Drama, Biography", 2020, "A chess prodigy rises through competitive chess while facing personal struggles.", "Anya Taylor-Joy, Thomas Brodie-Sangster", "Scott Frank"],
    ["Loki", "Web Series", "Sci-Fi, Fantasy, Adventure", 2021, "A mischievous time-traveling character encounters alternate timelines.", "Tom Hiddleston, Owen Wilson", "Michael Waldron"],
    ["The Last of Us", "Web Series", "Drama, Horror, Adventure", 2023, "Survivors travel across a devastated world after a fungal outbreak.", "Pedro Pascal, Bella Ramsey", "Craig Mazin"],
    ["Peaky Blinders", "Web Series", "Crime, Drama, History", 2013, "A gang leader expands his criminal empire in postwar Britain.", "Cillian Murphy, Paul Anderson", "Steven Knight"],
]

columns = ["title", "type", "genre", "year", "description", "cast", "director"]
movies = pd.DataFrame(movies_data, columns=columns)

movies["combined_features"] = (
    movies["genre"] + " " +
    movies["description"] + " " +
    movies["cast"] + " " +
    movies["director"] + " " +
    movies["type"]
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["combined_features"])
similarity_matrix = cosine_similarity(tfidf_matrix)

print("CineMatch AI: ML model trained successfully!")
print(f"Dataset contains {len(movies)} titles.")
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print("Algorithm: TF-IDF + Cosine Similarity")
display(movies[["title", "type", "genre", "year"]].head(10))



import gradio as gr

def recommend_movies(selected_title, genre_filter, number_of_recommendations):
    if not selected_title:
        return "Please select a movie first."

    selected_indices = movies.index[movies["title"] == selected_title].tolist()
    if not selected_indices:
        return "Selected title was not found."

    selected_idx = selected_indices[0]
    scores = similarity_matrix[selected_idx]

    results = movies.copy()
    results["similarity"] = scores
    results = results[results.index != selected_idx]

    if genre_filter != "All Genres":
        results = results[
            results["genre"].str.contains(genre_filter, case=False, na=False)
        ]

    results = results.sort_values("similarity", ascending=False).head(
        int(number_of_recommendations)
    )

    if results.empty:
        return "No matching titles found. Try another genre."

    cards = []
    for _, row in results.iterrows():
        score = row["similarity"] * 100
        cards.append(f"""
        <div class="movie-card">
            <div class="movie-type">{row['type']} · {int(row['year'])}</div>
            <h3>{row['title']}</h3>
            <div class="genre">{row['genre']}</div>
            <p>{row['description']}</p>
            <p><b>Director:</b> {row['director']}</p>
            <p><b>Cast:</b> {row['cast']}</p>
            <div class="score">
                Similarity score: {score:.1f}%
            </div>
        </div>
        """)
    return "\n".join(cards)


custom_css = """
body, .gradio-container {
    background: #0b0b12 !important;
    color: #f5f5f7 !important;
}
#main-title {
    text-align: center;
    color: #e8b86d;
}
#subtitle {
    text-align: center;
    color: #b8b8c8;
}
.movie-card {
    background: linear-gradient(145deg, #1c1c2b, #11111a);
    border: 1px solid #35354a;
    border-radius: 15px;
    padding: 18px;
    margin: 12px 0;
    color: #f5f5f7;
}
.movie-card h3 {
    color: #e8b86d;
    margin-bottom: 8px;
}
.movie-type {
    color: #a9a9c2;
    font-size: 13px;
}
.genre {
    color: #c7a5ff;
    margin-bottom: 10px;
}
.score {
    color: #76e0b5;
    font-weight: bold;
    border-top: 1px solid #35354a;
    padding-top: 10px;
}
"""

all_genres = sorted({
    genre.strip()
    for genres in movies["genre"]
    for genre in genres.split(",")
})

with gr.Blocks(css=custom_css, title="CineMatch AI") as demo:
    gr.Markdown(
        "# 🎬 CINE MATCH AI",
        elem_id="main-title"
    )
    gr.Markdown(
        "### Discover your next favorite movie or series",
        elem_id="subtitle"
    )
    gr.Markdown(
        "Choose a title and let our content-based ML model find similar stories."
    )

    with gr.Row():
        selected_title = gr.Dropdown(
            choices=movies["title"].tolist(),
            value="Inception",
            label="Choose a movie or web series"
        )
        genre_filter = gr.Dropdown(
            choices=["All Genres"] + all_genres,
            value="All Genres",
            label="Filter recommendations by genre"
        )
        number = gr.Slider(
            minimum=3,
            maximum=10,
            value=5,
            step=1,
            label="Number of recommendations"
        )

    recommend_button = gr.Button("✨ Find My Recommendations", variant="primary")
    output = gr.HTML(label="Your Recommendations")

    recommend_button.click(
        fn=recommend_movies,
        inputs=[selected_title, genre_filter, number],
        outputs=output
    )

    gr.Markdown(
        "---\n"
        "**How it works:** TF-IDF converts movie descriptions into numerical "
        "features. Cosine similarity ranks titles by their feature similarity.\n\n"
        "Built as an AI/ML mini project using Python, Pandas, Scikit-learn and Gradio."
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
