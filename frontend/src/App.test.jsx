import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

const item = (id, title, genres = ['Comedy'], extra = {}) => ({ item_id: id, title, genres, ...extra });

const TOY_STORY = item(1, 'Toy Story (1995)', ['Animation', 'Children'], { score: 0.91 });
const GODFATHER = item(2, 'Godfather, The (1972)', ['Crime', 'Drama'], { score: 0.8 });
const HEAT = item(3, 'Heat (1995)', ['Action'], { score: 0.7 });
const BUGS = item(4, "Bug's Life, A (1998)", ['Animation'], { score: 0.88 });

function reply(body, status = 200) {
  return { ok: status < 400, status, statusText: 'status', json: async () => body };
}

// Routes a request to a handler keyed by path (without the /api prefix).
function mockApi(overrides = {}) {
  const routes = {
    '/health': () => reply({ status: 'ok', users: 610, items: 9742 }),
    '/popular': () => reply({ items: [HEAT] }),
    '/recommend/1': () => reply({ user_id: 1, model: 'hybrid', personalised: true, items: [TOY_STORY, GODFATHER] }),
    '/users/1/history': () => reply({ user_id: 1, items: [item(9, 'Casablanca (1942)', ['Drama'], { rating: 5 })] }),
    '/items/1/similar': () => reply({ item_id: 1, model: 'hybrid', items: [BUGS] }),
    ...overrides,
  };
  const calls = [];
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url) => {
      const parsed = new URL(url, 'http://localhost');
      const path = parsed.pathname.replace(/^\/api/, '');
      calls.push({ path, params: Object.fromEntries(parsed.searchParams) });
      const handler = routes[path];
      return handler ? handler(parsed.searchParams) : reply({ detail: `no route ${path}` }, 404);
    })
  );
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

test('shows API status and the default user\'s recommendations', async () => {
  mockApi();
  render(<App />);

  expect(await screen.findByText('The Godfather')).toBeInTheDocument();
  expect(screen.getByText('Toy Story')).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Recommended for user 1' })).toBeInTheDocument();
  expect(screen.getByText('610 users · 9,742 items')).toBeInTheDocument();
  expect(screen.getByText('Heat')).toBeInTheDocument(); // popular section
  expect(screen.getByText('Casablanca')).toBeInTheDocument(); // history section
  expect(screen.getByText('★ 5.0')).toBeInTheDocument();
});

test('loads another user when the form is submitted', async () => {
  const calls = mockApi({
    '/recommend/7': () => reply({ user_id: 7, model: 'hybrid', personalised: true, items: [HEAT] }),
    '/users/7/history': () => reply({ user_id: 7, items: [] }),
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText('The Godfather');

  const input = screen.getByLabelText('User ID');
  await user.clear(input);
  await user.type(input, '7');
  await user.click(screen.getByRole('button', { name: 'Show recommendations' }));

  expect(await screen.findByRole('heading', { name: 'Recommended for user 7' })).toBeInTheDocument();
  expect(calls.some((call) => call.path === '/recommend/7')).toBe(true);
});

test('rejects an invalid user id without calling the API', async () => {
  const calls = mockApi();
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText('The Godfather');
  const before = calls.length;

  const input = screen.getByLabelText('User ID');
  await user.clear(input);
  await user.type(input, '0');
  await user.click(screen.getByRole('button', { name: 'Show recommendations' }));

  expect(screen.getByRole('alert')).toHaveTextContent('Enter a whole number of 1 or more.');
  expect(calls.length).toBe(before);
});

test('explains the cold start for a user with no history', async () => {
  mockApi({
    '/recommend/99999': () => reply({ user_id: 99999, model: 'popularity', personalised: false, items: [HEAT] }),
    '/users/99999/history': () => reply({ detail: 'User 99999 not found' }, 404),
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText('The Godfather');

  const input = screen.getByLabelText('User ID');
  await user.clear(input);
  await user.type(input, '99999');
  await user.click(screen.getByRole('button', { name: 'Show recommendations' }));

  expect(await screen.findByText(/cold start/)).toBeInTheDocument();
  expect(screen.getByRole('heading', { name: 'Popular right now' })).toBeInTheDocument();
  expect(screen.queryByText(/rated these highly/)).not.toBeInTheDocument();
});

test('switching the model asks the API for that model', async () => {
  const calls = mockApi({
    '/recommend/1': (params) =>
      reply({ user_id: 1, model: params.get('model'), personalised: true, items: [TOY_STORY] }),
  });
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText('Toy Story');

  await user.click(screen.getByRole('radio', { name: 'Content-based' }));

  await waitFor(() =>
    expect(calls.some((call) => call.path === '/recommend/1' && call.params.model === 'content')).toBe(true)
  );
  expect(screen.getByText(/genres and tags close/)).toBeInTheDocument();
});

test('selecting an item lists similar items and can be closed', async () => {
  const calls = mockApi();
  const user = userEvent.setup();
  render(<App />);
  expect(screen.getByText(/Select any item/)).toBeInTheDocument();

  await user.click(await screen.findByRole('button', { name: /Toy Story/ }));

  const aside = screen.getByLabelText('Similar items');
  expect(await within(aside).findByText("A Bug's Life")).toBeInTheDocument();
  expect(within(aside).getByRole('heading', { name: 'Similar to Toy Story' })).toBeInTheDocument();
  expect(calls.some((call) => call.path === '/items/1/similar' && call.params.k === '8')).toBe(true);

  await user.click(within(aside).getByRole('button', { name: 'Close' }));
  expect(screen.getByText(/Select any item/)).toBeInTheDocument();
});

test('shows a clear message when the API is offline', async () => {
  vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
  render(<App />);

  expect(await screen.findByText('API offline')).toBeInTheDocument();
  expect((await screen.findAllByRole('alert')).some((node) => /uvicorn app\.main:app/.test(node.textContent))).toBe(true);
});
