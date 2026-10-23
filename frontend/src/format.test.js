import { initials, posterHue, splitTitle } from './format';

describe('splitTitle', () => {
  it('separates the year', () => {
    expect(splitTitle('Toy Story (1995)')).toEqual({ name: 'Toy Story', year: '1995' });
  });

  it('moves a trailing article to the front', () => {
    expect(splitTitle('Godfather, The (1972)')).toEqual({ name: 'The Godfather', year: '1972' });
    expect(splitTitle('Bug\'s Life, A (1998)')).toEqual({ name: "A Bug's Life", year: '1998' });
  });

  it('keeps titles without a year', () => {
    expect(splitTitle('Untitled')).toEqual({ name: 'Untitled', year: null });
  });
});

describe('posterHue', () => {
  it('is based on the first genre and has a fallback', () => {
    expect(posterHue(['Sci-Fi', 'Drama'])).not.toBe(posterHue(['Romance']));
    expect(posterHue([])).toBe(210);
    expect(posterHue(['Unknown'])).toBe(210);
  });
});

describe('initials', () => {
  it('uses up to two words', () => {
    expect(initials('Toy Story')).toBe('TS');
    expect(initials('Heat')).toBe('H');
    expect(initials('The Sixth Sense')).toBe('SS');
    expect(initials('A')).toBe('A');
    expect(initials('!!!')).toBe('?');
  });
});
