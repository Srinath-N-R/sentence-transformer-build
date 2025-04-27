import regex as re
import torch
from enums import pat, CLS_ID


class Tokenizer():
    def __init__(self, path):
        '''
        We are initializing the merged-dict which we obtained after training the BPE tokenizer.
        We are also initializing variables we'll use for tokenization.
        '''
        with open(path, "r", encoding="utf-8") as f:
            old = f.read()
            old = eval(old)
            self.merged = {
                tuple(map(int, k.strip('()').split(','))): v
                for k, v in old.items()
            }
            self.rev_merged = {v:k for k, v in self.merged.items()}
        self.pat = pat


    def get_counts(self, list_of_tokens):
        '''
        Count the frequency of adjacent token pairs across a list of token sequences.
        '''
        counts = {}
    
        for token_list in list_of_tokens:
            for pair in zip(token_list, token_list[1:]):
                if pair not in counts:
                    counts[pair] = 1
                else:
                    counts[pair] += 1
        
        count_vals = [(v, k[0], k[1]) for k,v in counts.items()]
        count_vals = sorted(count_vals, reverse=True)
        return count_vals


    def merge_sequence(self, seq, tok_1, tok_2, new_token):
        '''
        Merge specific token pairs in a sequence into a single new token.
        Process:
            - Traverse the sequence.
            - Whenever a (tok_1, tok_2) pair is found, replace it with new_token.
            - Otherwise, retain the original token.
        '''

        merged_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == tok_1 and seq[i + 1] == tok_2:
                merged_seq.append(new_token)
                i += 2
            else:
                merged_seq.append(seq[i])
                i += 1
        return merged_seq


    def split_sequence(self, seq, merged_token):
        '''
        Split merged tokens in a sequence back into their original token pairs using the 'rev_merged' mapping.
        '''

        new_tokens = []
    
        for token in seq:
            if token == merged_token:
                tok_1, tok_2 = self.rev_merged[token]
                new_tokens.append(tok_1)
                new_tokens.append(tok_2)
            else:
                new_tokens.append(token)
    
        return new_tokens
    

    def num_to_str(self, number):
        '''
        Convert an integer (0–255) to its corresponding single UTF-8 character.
        '''
        byte_representation = number.to_bytes(1, byteorder='big')
        utf8_string = byte_representation.decode('utf-8')
        return utf8_string

    
    def encode(self, text):
        '''
        Encode a text sequence by iteratively merging token pairs based on a predefined merge map.

        Process:
            - Split the input text into fragments.
            - Tokenize each fragment into byte-encoded tokens.
            - Iteratively merge token pairs found in the 'merged' mapping, 
            prioritizing pairs with the smallest assigned merged token value.
            - Continue merging until no more applicable pairs are found.

        Notes:
            Priority is given to token pairs with the lowest merged value to preserve merge coherence.
        '''

        texts = re.findall(self.pat, text)
        list_of_tokens = [list(text.encode('utf-8')) for text in texts]
    
        seen = set()
        while True:
            count_vals = self.get_counts(list_of_tokens)
            pairs = [(val[1:]) for val in count_vals]
            pairs = [pair for pair in pairs if pair in self.merged and pair not in seen]
            valid_pairs = {pair:self.merged[pair] for pair in pairs}
            
            if valid_pairs:
                min_pair = min(valid_pairs, key=lambda x:self.merged[x])
                seen.add(min_pair)
                tok_1, tok_2 = min_pair
                new_token = self.merged[(tok_1, tok_2)]
    
                new_list_of_tokens = []
                for tokens in list_of_tokens:
                    new_tokens = self.merge_sequence(tokens, tok_1, tok_2, new_token)
                    new_list_of_tokens.append(new_tokens)
                list_of_tokens = new_list_of_tokens
            else:
                break
    
        return list_of_tokens


    def decode(self, list_of_tokens):
        '''
        Decode a list of merged token sequences back into the original text string.
        
        Process:
            - Iteratively split merged tokens into their original token pairs, 
            starting from the highest merged token value.
            - Flatten the final list of tokens.
            - Convert each token back to its corresponding UTF-8 character.
            - Join the characters into a single decoded string.
        
        Notes:
            Priority is given to token pairs with the highest merged value to preserve un-merge coherence.
        '''

        while True:
            valid_tokens = []
            for tokens in list_of_tokens:
                valid_tokens.extend([token for token in tokens if token in self.rev_merged])
    
            if valid_tokens:
                max_token = max(valid_tokens)                 
                new_list_of_tokens = []
                for tokens in list_of_tokens:
                    new_tokens = self.split_sequence(tokens, max_token)
                    new_list_of_tokens.append(new_tokens)
                list_of_tokens = new_list_of_tokens
            else:
                break

        final_tokens = []
        for tokens in list_of_tokens:
            final_tokens.extend(tokens)
        str_tokens = [self.num_to_str(token) for token in final_tokens]
        return "".join(str_tokens)

    def tokenize(self, text):
        '''
        Tokenize input text into a sequence of token IDs suitable for Sentence Tranformer.

        Process:
            - Encode the text into a list of merged token sequences.
            - Flatten the list into a single sequence of token IDs.
            - Prepend the special CLS_ID token to the sequence.
            - Return the result as a PyTorch tensor of type torch.long.

        '''

        list_of_tokens = self.encode(text)
        ids = [tok for toks in list_of_tokens for tok in toks]
        ids = [CLS_ID] + ids
        return torch.tensor(ids, dtype=torch.long)
    

