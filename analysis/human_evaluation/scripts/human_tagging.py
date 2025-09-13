#!/usr/bin/env python3
import json
import os
import time
import textwrap
import datetime
from pathlib import Path
import webbrowser
import platform
import random
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich import box
from rich.style import Style

# Constants
ALLOWED_TAGS = {
    "Emotional_Tone": ["Joy", "Sadness", "Anger", "Bittersweet", "Nostalgia", "Euphoria", "Triumph", "Regret", "Loneliness", "None"],
    "Thematic_Content": ["Love", "Heartbreak", "Wealth", "Success", "Struggle", "Party", "Protest", "Spirituality", "None"],
    "Narrative_Structure": ["Narrative_Yes", "Narrative_No", "Conflict_Resolution", "First_Person", "Third_Person", "None"],
    "Lyrical_Style": ["Repetition", "Metaphor", "Alliteration", "Vivid_Imagery", "Rhetorical_Question", "None"]
}

# Tag descriptions for help
TAG_DESCRIPTIONS = {
    "Joy": "Expressions of happiness, delight, or celebration",
    "Sadness": "Expressions of sorrow, grief, or melancholy",
    "Anger": "Expressions of rage, frustration, or indignation",
    "Bittersweet": "Mixed emotions of happiness and sadness",
    "Nostalgia": "Longing for the past or fond memories",
    "Euphoria": "Intense happiness or elation",
    "Triumph": "Celebration of victory or overcoming adversity",
    "Regret": "Expressions of remorse or wishing things were different",
    "Loneliness": "Feelings of isolation or being alone",
    
    "Love": "Romantic or deep affection themes",
    "Heartbreak": "Themes of romantic loss or emotional pain",
    "Wealth": "References to money, luxury, or material success",
    "Success": "Themes of achievement or accomplishment",
    "Struggle": "Dealing with hardship or obstacles",
    "Party": "Celebration, dancing, or social gathering themes",
    "Protest": "Political or social justice themes",
    "Spirituality": "References to faith, religion, or spiritual concepts",
    
    "Narrative_Yes": "Song tells a story with characters or events",
    "Narrative_No": "Song doesn't tell a specific story",
    "Conflict_Resolution": "Story presents a problem and resolution",
    "First_Person": "Lyrics primarily use 'I' or 'we' perspective",
    "Third_Person": "Lyrics primarily tell someone else's story",
    
    "Repetition": "Repeated phrases or choruses are prominent",
    "Metaphor": "Uses comparisons to convey meaning indirectly",
    "Alliteration": "Repeated consonant sounds at the beginning of words",
    "Vivid_Imagery": "Creates strong visual or sensory descriptions",
    "Rhetorical_Question": "Asks questions not meant to be answered",
    
    "None": "This category doesn't apply to this song"
}

# Color themes for different tag categories
TAG_COLORS = {
    "Emotional_Tone": "magenta",
    "Thematic_Content": "blue",
    "Narrative_Structure": "green",
    "Lyrical_Style": "yellow"
}

# Project metadata
PROJECT_INFO = {
    "name": "ResoNote: A Semantic Playlist Generator for Emotionally-Aligned Music Recommendation",
    "researcher": "Druhin Tarafder",
    "institution": "University of London, Computer Science",
    "contact": "dt158@student.london.ac.uk",
    "version": "1.0.0",
    "date": "2023-2024"
}

# Path to dev.json file - hardcoded to avoid asking the annotator
DEV_FILE_PATH = "analysis/llm_selection/splits/dev.json"

# Create console
console = Console()

def clear_screen():
    """Clear the terminal screen based on OS."""
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")

def display_welcome():
    """Display an attractive welcome screen."""
    clear_screen()
    
    # ASCII art title
    title = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║   ██████╗ ███████╗███████╗ ██████╗ ███╗   ██╗ ██████╗ ████████╗███████╗  ║
    ║   ██╔══██╗██╔════╝██╔════╝██╔═══██╗████╗  ██║██╔═══██╗╚══██╔══╝██╔════╝  ║
    ║   ██████╔╝█████╗  ███████╗██║   ██║██╔██╗ ██║██║   ██║   ██║   █████╗    ║
    ║   ██╔══██╗██╔══╝  ╚════██║██║   ██║██║╚██╗██║██║   ██║   ██║   ██╔══╝    ║
    ║   ██║  ██║███████╗███████║╚██████╔╝██║ ╚████║╚██████╔╝   ██║   ███████╗  ║
    ║   ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝    ╚═╝   ╚══════╝  ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """
    
    console.print(Panel.fit(title, border_style="bright_blue"))
    
    # Description
    description = f"""
    Welcome to the Music Tagging System for the ResoNote project.
    
    This program will guide you through the process of listening to and tagging music tracks
    based on their emotional tone, thematic content, narrative structure, and lyrical style.
    
    Your expert evaluation will help us benchmark AI performance in understanding music lyrics
    and develop more emotionally-aligned music recommendation systems.
    
    [bold]Researcher:[/] {PROJECT_INFO['researcher']}
    [bold]Institution:[/] {PROJECT_INFO['institution']}
    [bold]Contact:[/] {PROJECT_INFO['contact']}
    """
    
    console.print(Panel(description, title="About This Study", border_style="green"))
    
    # Continue prompt
    console.print("\n[bold cyan]Press Enter to continue...[/]", end="")
    input()

def collect_annotator_info():
    """Collect information about the annotator."""
    clear_screen()
    
    console.print(Panel("Please provide some information about yourself", title="Annotator Profile", border_style="yellow"))
    
    name = Prompt.ask("[bold]Your Name[/]")
    occupation = Prompt.ask("[bold]Your Occupation[/]")
    
    # Experience level
    experience_table = Table(box=box.ROUNDED)
    experience_table.add_column("Level", style="cyan")
    experience_table.add_column("Description")
    experience_table.add_row("1", "Casual listener")
    experience_table.add_row("2", "Music enthusiast")
    experience_table.add_row("3", "Amateur musician/performer")
    experience_table.add_row("4", "Music educator")
    experience_table.add_row("5", "Professional musician/industry expert")
    
    console.print(experience_table)
    
    experience = Prompt.ask(
        "[bold]Your Music Expertise Level[/] (1-5)", 
        choices=["1", "2", "3", "4", "5"]
    )
    
    genres = Prompt.ask("[bold]Music Genres You're Most Familiar With[/] (comma-separated)")
    
    # Confirm information
    info = {
        "name": name,
        "occupation": occupation,
        "expertise_level": int(experience),
        "familiar_genres": [g.strip() for g in genres.split(",")],
        "session_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    clear_screen()
    
    # Display summary
    summary = Table(title="Annotator Profile", box=box.ROUNDED, border_style="yellow")
    summary.add_column("Field", style="bold cyan")
    summary.add_column("Value")
    
    # Only show relevant info to user (no technical details)
    summary.add_row("Name", info["name"])
    summary.add_row("Occupation", info["occupation"])
    summary.add_row("Music Expertise Level", str(info["expertise_level"]))
    summary.add_row("Familiar Genres", ", ".join(info["familiar_genres"]))
    
    console.print(summary)
    
    if Confirm.ask("[bold]Is this information correct?[/]"):
        return info
    else:
        return collect_annotator_info()

def show_instructions():
    """Display instructions for the tagging process."""
    clear_screen()
    
    # Show instructions as rich text panels instead of markdown
    console.print(Panel("Tagging Instructions", border_style="cyan", title="Instructions"))
    
    console.print("\n[bold]For each track, you'll follow these steps:[/]")
    console.print("  1. Read the track information and lyrics")
    console.print("  2. Listen to the track on Spotify (link will open automatically)")
    console.print("  3. Tag the track by answering questions about its content")
    
    console.print("\n[bold]Tagging Categories:[/]")
    console.print("  • [bold magenta]Emotional Tone[/]: The feelings or emotions expressed")
    console.print("  • [bold blue]Thematic Content[/]: The subject matter or topics")
    console.print("  • [bold green]Narrative Structure[/]: How the song tells its story")
    console.print("  • [bold yellow]Lyrical Style[/]: The literary techniques used")
    
    console.print("\n[bold]Tips for Effective Tagging:[/]")
    console.print("  • Listen to each track completely before tagging")
    console.print("  • Consider both the music and lyrics when making judgments")
    console.print("  • Trust your expertise and intuition")
    console.print("  • Use the 'None' option only when no other tags apply")
    console.print("  • You can select multiple tags in each category")
    console.print("  • Rate your confidence on a scale of 1-10 for each tag")
    
    console.print("\n[bold]Timing:[/]")
    console.print("  • Take your time! There's no rush")
    console.print("  • You can pause between tracks if needed")
    console.print("  • Your progress will be saved after each track")
    
    console.print("\n[bold cyan]Press Enter when you're ready to begin tagging...[/]", end="")
    input()

def format_lyrics(lyrics):
    """Format lyrics for better readability."""
    if not lyrics:
        return Text("[No lyrics available]", style="italic")
    
    # Replace newlines with proper line breaks
    lines = lyrics.split("\n")
    
    # Wrap long lines
    wrapped_lines = []
    for line in lines:
        if len(line.strip()) == 0:
            wrapped_lines.append("")
        else:
            wrapped_parts = textwrap.wrap(line, width=80)
            wrapped_lines.extend(wrapped_parts)
    
    # Format with proper spacing
    formatted = Text("\n".join(wrapped_lines))
    return formatted

def open_spotify_url(track_id):
    """Open Spotify URL in the default browser."""
    url = f"https://open.spotify.com/track/{track_id}"
    
    # Create a visually appealing button-like link
    link_display = Text("  🎧 LISTEN ON SPOTIFY  ", style="white on green")
    
    console.print("\n")
    console.print(Panel(link_display, border_style="green"))
    console.print(f"[dim]URL: {url}[/dim]")
    
    try:
        webbrowser.open(url)
        return url
    except Exception:
        console.print(f"[bold red]Could not open browser automatically.[/]")
        console.print(f"[yellow]Please manually copy and paste this URL: {url}[/]")
        return url

def display_tag_help(category):
    """Display help information for tags in a specific category."""
    clear_screen()
    
    console.print(Panel(f"Help: {category} Tags", border_style=TAG_COLORS[category]))
    
    help_table = Table(box=box.ROUNDED)
    help_table.add_column("Tag", style=f"bold {TAG_COLORS[category]}")
    help_table.add_column("Description")
    
    for tag in ALLOWED_TAGS[category]:
        help_table.add_row(tag, TAG_DESCRIPTIONS.get(tag, "No description available"))
    
    console.print(help_table)
    console.print("\n[bold cyan]Press Enter to return to tagging...[/]", end="")
    input()

def display_track_info(track, track_number, total_tracks):
    """Display track information and lyrics."""
    clear_screen()
    
    # Header with progress
    console.print(Panel(
        f"Track {track_number} of {total_tracks}",
        title="Now Tagging",
        border_style="blue"
    ))
    
    # Track info panel
    info_panel = Panel(
        Text.assemble(
            ("Track: ", "bold cyan"), (f"{track.get('track_name', 'Unknown')}\n", "white"),
            ("Artist: ", "bold cyan"), (f"{track.get('artist_name', 'Unknown')}\n", "white"),
            ("\nAudio Features:\n", "bold cyan"),
            ("Energy: ", "dim cyan"), (f"{track.get('energy', 'N/A')}\n", "white"),
            ("Valence: ", "dim cyan"), (f"{track.get('valence', 'N/A')}\n", "white"),
            ("Tempo: ", "dim cyan"), (f"{track.get('tempo', 'N/A')} BPM\n", "white"),
        ),
        title="Track Information",
        border_style="green"
    )
    
    # Lyrics panel
    lyrics_panel = Panel(
        format_lyrics(track.get("lyrics", "")),
        title="Lyrics",
        border_style="yellow",
        width=100
    )
    
    console.print(info_panel)
    console.print(lyrics_panel)
    
    # Open Spotify
    spotify_url = open_spotify_url(track.get("track_id", ""))
    
    console.print("\n[bold cyan]Press Enter after listening to continue with tagging...[/]", end="")
    input()
    
    return spotify_url

def tag_track(track, track_number, total_tracks):
    """Interactive tagging function for a single track."""
    track_start_time = time.time()
    
    # First display track info and open Spotify
    spotify_url = display_track_info(track, track_number, total_tracks)
    
    while True:  # Loop to allow retrying the same track
        # Collect tags
        tags = {}
        tag_start_time = time.time()
        retry_track = False
        
        # Loop through each category for tagging
        for category, allowed_values in ALLOWED_TAGS.items():
            clear_screen()
            
            # Display category info
            console.print(Panel(
                f"Now tagging [bold]{category}[/] for track: {track.get('track_name', 'Unknown')}",
                border_style=TAG_COLORS[category]
            ))
            
            # Show help option
            console.print(f"[dim]Type 'help' for explanations of {category} tags[/dim]\n")
            
            facet_tags = []
            
            # Loop through tags
            for tag in allowed_values:
                if tag == "None":
                    # Skip asking about None, we'll add it if no other tags were selected
                    continue
                
                # Ask about this tag using single character input
                response = Prompt.ask(
                    f"Does this track express [bold {TAG_COLORS[category]}]{tag}[/]?",
                    choices=["y", "n", "help", "r"]  # 'r' for retry instead of 'retry'
                )
                
                if response.lower() == "help":
                    display_tag_help(category)
                    # After showing help, restart this tag
                    clear_screen()
                    console.print(Panel(
                        f"Now tagging [bold]{category}[/] for track: {track.get('track_name', 'Unknown')}",
                        border_style=TAG_COLORS[category]
                    ))
                    console.print(f"[dim]Type 'help' for explanations of {category} tags[/dim]\n")
                    
                    # Ask about this tag again
                    response = Prompt.ask(
                        f"Does this track express [bold {TAG_COLORS[category]}]{tag}[/]?",
                        choices=["y", "n", "r"]  # 'r' for retry
                    )
                
                if response.lower() == "r":
                    # Restart the entire track
                    retry_track = True
                    break
                
                if response.lower() == "y":
                    # Ask for confidence score (1-10 scale)
                    confidence = Prompt.ask(
                        f"How confident are you about [bold]{tag}[/]?",
                        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                    )
                    
                    # Map 1-10 scale to 0.1-1.0
                    score = int(confidence) * 0.1
                    facet_tags.append({"tag": tag, "score": score})
            
            # If user chose to retry, break out of category loop
            if retry_track:
                break
                
            # If no tags were selected, use None
            if not facet_tags:
                facet_tags.append({"tag": "None", "score": 1.0})
                
            tags[category] = facet_tags
        
        # If we're retrying the track, continue to next iteration of while loop
        if retry_track:
            # Show track info again
            spotify_url = display_track_info(track, track_number, total_tracks)
            continue
            
        # Calculate tagging time
        tag_time = time.time() - tag_start_time
        
        # Show summary of tags
        clear_screen()
        console.print(Panel(f"Summary for: {track.get('track_name', 'Unknown')}", border_style="cyan"))
        
        summary_table = Table(box=box.ROUNDED)
        summary_table.add_column("Category", style="bold")
        summary_table.add_column("Tags")
        
        for category, tag_list in tags.items():
            tag_text = ", ".join([f"{t['tag']} ({t['score']:.1f})" for t in tag_list])
            summary_table.add_row(category, tag_text)
        
        console.print(summary_table)
        
        # Ask if they want to save or retry using single character input
        choice = Prompt.ask(
            "What would you like to do?",
            choices=["s", "r"]  # 's' for save, 'r' for retry
        )
        
        if choice == "s":
            # Calculate total time for this track
            total_track_time = time.time() - track_start_time
            
            # Return the tagged result with timing information
            return {
                "track_id": track.get("track_id", ""),
                "track_name": track.get("track_name", ""),
                "artist_name": track.get("artist_name", ""),
                "spotify_url": spotify_url,
                "tags": tags,
                "tagging_time_seconds": round(tag_time, 2),
                "total_time_seconds": round(total_track_time, 2)
            }
        
        # If retry, show track info again and restart tagging
        spotify_url = display_track_info(track, track_number, total_tracks)

def save_progress(tagged_tracks, annotator_info, output_dir):
    """Save current progress to file."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare complete data with metadata - moved project to global level
    complete_data = {
        "annotator": annotator_info,
        "tracks": tagged_tracks,
        "session_info": {
            "completed_tracks": len(tagged_tracks),
            "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "average_time_per_track": round(sum(t.get("total_time_seconds", 0) for t in tagged_tracks) / max(1, len(tagged_tracks)), 2)
        }
    }
    
    # Save to file
    output_file = output_dir / "tagged_tracks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(complete_data, f, indent=2, ensure_ascii=False)
    
    return output_file

def show_break_screen():
    """Show a break screen with a random motivational message."""
    clear_screen()
    
    messages = [
        "Great job! Take a moment to rest your ears.",
        "Your expert insights are invaluable to this research.",
        "Your careful listening makes this research possible!",
        "Music analysis is both art and science - thank you for bringing both.",
        "Your tagging helps us understand how AI interprets music!",
    ]
    
    console.print(Panel(
        random.choice(messages),
        title="Time for a Break!",
        border_style="green"
    ))
    
    console.print("\n[bold cyan]Press Enter when you're ready to continue...[/]", end="")
    input()

def show_completion_stats(tagged_tracks, annotator_info):
    """Show completion statistics without timing information."""
    clear_screen()
    
    # Count tags
    tag_counts = {category: {} for category in ALLOWED_TAGS.keys()}
    for track in tagged_tracks:
        for category, tags in track.get("tags", {}).items():
            for tag_info in tags:
                tag = tag_info.get("tag")
                if tag:
                    tag_counts[category][tag] = tag_counts[category].get(tag, 0) + 1
    
    # Create tag frequency tables
    tag_tables = []
    for category in ALLOWED_TAGS.keys():
        table = Table(title=f"{category} Tag Frequencies", box=box.ROUNDED)
        table.add_column("Tag", style=TAG_COLORS[category])
        table.add_column("Count")
        table.add_column("Percentage")
        
        for tag in ALLOWED_TAGS[category]:
            count = tag_counts[category].get(tag, 0)
            percentage = (count / len(tagged_tracks)) * 100
            table.add_row(tag, str(count), f"{percentage:.1f}%")
        
        tag_tables.append(table)
    
    # Display all tables
    console.print(Panel(
        f"[bold green]Congratulations, {annotator_info['name']}![/]\n\nYou've completed tagging all tracks!",
        border_style="green"
    ))
    
    for table in tag_tables:
        console.print(table)
    
    console.print("\n[bold cyan]Thank you for your valuable contribution to this research![/]")

def load_existing_progress(output_dir):
    """Try to load existing progress."""
    output_file = output_dir / "tagged_tracks.json"
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("annotator", {}), data.get("tracks", [])
        except Exception:
            return None, []
    return None, []

def create_project_info_file(project_dir):
    """Create a project info file to store global project metadata."""
    project_file = project_dir / "project_info.json"
    
    # Only create if it doesn't exist
    if not project_file.exists():
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(PROJECT_INFO, f, indent=2, ensure_ascii=False)

def main():
    """Main function to run the tagging program."""
    total_start_time = time.time()
    
    # Display welcome screen
    display_welcome()
    
    # Collect annotator information
    annotator_info = collect_annotator_info()
    
    # Create project directory and save project info
    project_dir = Path("analysis/human_evaluation")
    project_dir.mkdir(parents=True, exist_ok=True)
    create_project_info_file(project_dir)
    
    # Create output directory path with new structure
    output_dir = Path(f"analysis/human_evaluation/results/{annotator_info['name'].replace(' ', '_')}")
    
    # Check for existing progress
    existing_info, existing_tracks = load_existing_progress(output_dir)
    
    if existing_tracks:
        console.print(Panel(
            f"Found existing progress: {len(existing_tracks)} tracks already tagged.",
            title="Resume Session",
            border_style="yellow"
        ))
        
        if Confirm.ask("Would you like to resume from where you left off?"):
            if existing_info is not None:
                annotator_info = existing_info
                tagged_tracks = existing_tracks
                console.print(f"[green]Resuming with {len(tagged_tracks)} tracks already completed.[/]")
            else:
                console.print("[yellow]Existing annotator info is invalid. Using your new information.[/]")
                tagged_tracks = existing_tracks
        else:
            if Confirm.ask("[bold red]This will overwrite your previous work. Are you sure?"):
                tagged_tracks = []
            else:
                tagged_tracks = existing_tracks
    else:
        tagged_tracks = []
    
    # Show instructions
    show_instructions()
    
    # Load the dev tracks (the 50 track sample) - hardcoded path
    try:
        with open(DEV_FILE_PATH, 'r', encoding='utf-8') as f:
            all_tracks = json.load(f)
    except Exception:
        console.print(f"[bold red]Error loading tracks. Please contact the research team.[/]")
        return
    
    # Filter out already tagged tracks
    tagged_ids = {track["track_id"] for track in tagged_tracks}
    tracks_to_tag = [track for track in all_tracks if track.get("track_id") not in tagged_ids]
    
    if not tracks_to_tag:
        console.print("[green]You have already tagged all tracks![/]")
        show_completion_stats(tagged_tracks, annotator_info)
        return
    
    console.print(f"[green]You have {len(tracks_to_tag)} tracks left to tag.[/]")
    console.print("[bold cyan]Press Enter to begin tagging...[/]", end="")
    input()
    
    # Process each track one by one
    for i, track in enumerate(tracks_to_tag, 1):
        # Show break reminder every 10 tracks
        if i > 1 and (i-1) % 10 == 0:
            show_break_screen()
        
        # Tag the track
        tagged = tag_track(track, i + len(tagged_tracks), len(all_tracks))
        tagged_tracks.append(tagged)
        
        # Save after each track in case of interruption
        output_file = save_progress(tagged_tracks, annotator_info, output_dir)
        
        # Show progress
        clear_screen()
        console.print(f"[green]Track {i} of {len(tracks_to_tag)} completed.[/]")
        console.print(f"[green]Progress: {i}/{len(tracks_to_tag)} tracks ({int(i/len(tracks_to_tag)*100)}%)[/]")
        
        if i < len(tracks_to_tag):
            console.print("\n[bold cyan]Press Enter to continue to the next track...[/]", end="")
            input()
    
    # Show completion stats without timing information
    show_completion_stats(tagged_tracks, annotator_info)
    
    # Final save
    output_file = save_progress(tagged_tracks, annotator_info, output_dir)
    console.print(f"\n[green]Results saved successfully![/]")
    
    # Also save a summary file for easy access (not shown to user)
    summary_dir = Path("analysis/human_evaluation/summaries")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_data = {
        "annotator": annotator_info,
        "completion_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tracks_completed": len(tagged_tracks),
        "total_time_minutes": round((time.time() - total_start_time) / 60, 2),
        "result_path": str(output_file)
    }
    
    summary_file = summary_dir / f"{annotator_info['name'].replace(' ', '_')}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted. Progress has been saved.[/]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred: {e}[/]")
        console.print("[yellow]Please contact the research team for assistance.[/]")