

# Essentially parses out the intial info left on text files saved on gutenberg
def gutenberg_parser(filename: object) -> object:
    text = []
    start_text = False
    with open(filename, 'r') as f:
        for lineindex, line in enumerate(f,start=1):
            if start_text:
                text.append(line.split())

            if line.startswith('*** S'):
                start_text = True
            if line.startswith('*** E'):
                break
                #text = self.clean_text(raw_text)
    all_text = sum(text, [])
    return all_text


if __name__ == '__main__':
    print(gutenberg_parser('/Users/billywallace/Desktop/Movie-Critic-NLP/books/Dracula.txt'))
