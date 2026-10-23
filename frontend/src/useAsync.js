import { useEffect, useState } from 'react';

/**
 * Runs `load(signal)` whenever `deps` change and tracks { data, error, loading }.
 * The previous request is aborted, so a slow answer can never overwrite a newer one.
 */
export function useAsync(load, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    const controller = new AbortController();
    setState((previous) => ({ ...previous, error: null, loading: true }));

    load(controller.signal)
      .then((data) => setState({ data, error: null, loading: false }))
      .catch((error) => {
        if (error.name === 'AbortError') return;
        setState({ data: null, error, loading: false });
      });

    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
