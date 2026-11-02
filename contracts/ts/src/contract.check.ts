// Compiled by `npm run check`, never run: it pins what the generated types promise the site.
import { createApi } from "./index";

const api = createApi("http://localhost:8000");

export async function smoke() {
  const health = await api.GET("/v1/health");
  const status: "ok" | undefined = health.data?.status;

  const slot = { slot_id: "00000000-0000-4000-8000-000000000001" };
  await api.POST("/v1/holds", { body: slot, params: { header: { "Idempotency-Key": "abcdefgh-1" } } });
  // @ts-expect-error a hold needs its Idempotency-Key
  await api.POST("/v1/holds", { body: slot });

  const failure = await api.GET("/v1/analyses/{analysis_id}", {
    params: { path: { analysis_id: slot.slot_id } },
  });
  const code: string | undefined = failure.error?.code; // every failure is an ErrorResponse
  return { status, code };
}
