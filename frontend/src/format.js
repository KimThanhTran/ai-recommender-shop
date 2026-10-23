const GENRE_HUE = {
  Action: 12,
  Adventure: 35,
  Animation: 290,
  Children: 320,
  Comedy: 48,
  Crime: 0,
  Documentary: 170,
  Drama: 215,
  Fantasy: 265,
  'Film-Noir': 230,
  Horror: 350,
  Musical: 300,
  Mystery: 250,
  Romance: 335,
  'Sci-Fi': 190,
  Thriller: 5,
  War: 100,
  Western: 25,
  IMAX: 200,
};

/** Splits "Godfather, The (1972)" into { name: "The Godfather", year: "1972" }. */
export function splitTitle(title) {
  const match = /^(.*?)\s*\((\d{4})\)\s*$/.exec(title);
  let name = match ? match[1] : title;
  const year = match ? match[2] : null;

  const article = /^(.*), (The|A|An)$/.exec(name);
  if (article) name = `${article[2]} ${article[1]}`;
  return { name, year };
}

/** Poster colour from the first genre; the dataset has no images, so cards use a tinted placeholder. */
export function posterHue(genres) {
  return GENRE_HUE[genres?.[0]] ?? 210;
}

export function initials(name) {
  const letters = name
    .replace(/^(The|A|An) (?=\S)/, '') // "The Godfather" -> "G", not "TG"
    .replace(/[^\p{L}\p{N} ]/gu, '')
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0]);
  return letters.join('').toUpperCase() || '?';
}
