import { ItemCard } from './ItemCard';

/** Renders loading skeletons, an error, an empty message or the cards, depending on the request state. */
export function ItemGrid({
  items,
  loading,
  error,
  onSelect,
  selectedId,
  showScore = false,
  emptyText = 'Nothing to show.',
  skeletons = 6,
}) {
  if (error) {
    return (
      <p className="message message-error" role="alert">
        Could not load this section: {error.message}
      </p>
    );
  }
  if (loading && !items?.length) {
    return (
      <div className="grid" aria-busy="true" aria-label="Loading">
        {Array.from({ length: skeletons }, (_, index) => (
          <div className="card skeleton" key={index} />
        ))}
      </div>
    );
  }
  if (!items?.length) return <p className="message">{emptyText}</p>;

  return (
    <div className={`grid${loading ? ' grid-stale' : ''}`} aria-busy={loading}>
      {items.map((item) => (
        <ItemCard
          key={item.item_id}
          item={item}
          showScore={showScore}
          selected={item.item_id === selectedId}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}
