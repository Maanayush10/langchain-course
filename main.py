from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    print("Hello from Langchain-COURSE! This is AYUSHMAN.")

    information = """ Batman is a superhero who appears in American comic books published by DC Comics. Batman was created by writer Bill Finger and artist Bob Kane, and debuted in the 27th issue of the comic book Detective Comics on March 30, 1939. In the DC Universe, Batman is the alias of Bruce Wayne, a wealthy American playboy, philanthropist, and industrialist who resides in the fictional Gotham City. Originally a millionaire, the character is later depicted as a billionaire. As Batman, he vows never to kill any criminal.[4] His origin story features him swearing vengeance against criminals after witnessing the murder of his parents, Thomas and Martha, as a child, a vendetta tempered by the ideal of justice. Vowing never to use firearms and other lethal weaponry, he trains himself physically and intellectually, crafts a bat-inspired persona and arsenal, utilizes his family wealth and resources, and monitors the Gotham streets at night. He lives at his personal residence, Wayne Manor, which also houses his secret base, Batcave. Kane, Finger, and other creators accompanied Batman with supporting characters, including his sidekicks Robin and Batgirl; allies Alfred Pennyworth and James Gordon; love interest and occasional adversary Catwoman; as well as foes such as the Penguin, the Riddler, Two-Face, and his archenemy, the Joker.

    Kane conceived Batman in early 1939 to capitalize on the popularity of Superman; although Kane frequently claimed sole creation credit, Finger substantially developed the concept from a generic superhero into something more bat-like. They drew inspiration from pulp fiction characters such as Sherlock Holmes and the Shadow. Batman received a spin-off publication, Batman, in 1940. Kane and Finger introduced Batman as a ruthless vigilante who frequently killed or maimed criminals, but he evolved into a just, tempered superhero with a stringent moral code that prohibits killing during the 1940s. Unlike most superheroes, Batman does not possess any superpowers, instead relying on his intellect, fighting skills, and wealth. The 1960s Batman television series used a camp aesthetic, which continued to be associated with Batman for years after it ended. Various creators worked to return Batman to his darker roots in the 1970s and 1980s, culminating with the 1986 miniseries The Dark Knight Returns by Frank Miller.

    DC has featured Batman in many comic books, including comics published under its imprints such as Vertigo and Black Label; he has been considered DC's flagship character[5][6] since the 1990s. The longest-running Batman comic, Detective Comics, is the longest-running comic book in the United States. Batman is frequently depicted alongside other DC superheroes, such as Superman and Wonder Woman, as a member of organizations such as the Justice League and the Outsiders. In addition to Bruce Wayne, other characters used the Batman persona, such as Jean-Paul Valley / Azrael in the 1993–1994 "Knightfall" story arc; Dick Grayson, the first Robin, from 2009 to 2011; and Jace Fox, the son of Wayne's ally Lucius, since 2021.[7] DC has also published comics featuring alternate versions of Batman, including the incarnation seen in The Dark Knight Returns and its successors, the incarnation from the Flashpoint (2011) event, and numerous interpretations in comics published under the Elseworlds label.

    Batman is one of the most iconic characters in popular culture and has been listed among the greatest comic book superheroes and characters ever created. He is one of the most commercially successful superheroes, the second best-selling comic book series in history with 460 million copies sold worldwide,[8] and his likeness has been licensed and featured in various media and merchandise sold around the world; this includes toy lines such as Lego Batman and video games such as the Batman: Arkham series. Batman has been adapted in many live-action and animated television series and films. Adam West portrayed him in the 1960s Batman television series, and he has been portrayed in films by Michael Keaton, Val Kilmer, George Clooney, Christian Bale, Ben Affleck, and Robert Pattinson. Many actors, most prolifically Kevin Conroy, have provided Batman's voice in animation and video games. In September 2024, Batman was given a star on the Hollywood Walk of Fame, being the first superhero to receive the honor."""

    summary_template = (
        "Given the information {information} about a fictional charecter I want you to create : "
        "1) A short summary "
        "2)Two interesting facts about them."
    )

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
