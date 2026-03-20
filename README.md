## To acquire API keys:

- [Google AI Studio](https://aistudio.google.com/api-keys)
- [Pinecone](https://app.pinecone.io/organizations/keys)
- [OpenRouter](https://openrouter.ai/)

## How this was created

- based on [this video](https://www.youtube.com/watch?v=hem5D1uvy-w)

- claude prompt <plan mode>

```
/plugin install frontend-design@claude-code-plugins
```

```
I want to use gemini's new embeddings model (https://ai.google.dev/gemini-api/docs/embeddings) in order to have a pinecone vector database filled with videos, images & text.
build a plan to set all this up
create me a .env.example file with placeholders and I will drop in:
- my pinecone api key
- my gemini api key
- my openrouter api key
```

- _implement the plan above_
- claude prompt

```
media has been dropped in.  ingest that into pinecone, then build me a simple chat web app on localhost so I can test that everything works well.  I want to use sonnet as the model for chat interaction.  Make sure you use the front end design skill to build this chat web app.  just keep it super simple
```

```
Update the chat app to display images/videos/pdfs inline when they come back as sources
```
