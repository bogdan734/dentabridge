import createClient, { type ClientOptions } from "openapi-fetch";
import type { paths } from "./schema";

export type { components, operations, paths } from "./schema";

/**
 * A typed client for the DentaBridge API. `baseUrl` is the API's origin: the paths already carry
 * `/v1`. Requests send no credentials; the API allows listed origins only.
 */
export function createApi(baseUrl: string, options: Omit<ClientOptions, "baseUrl"> = {}) {
  return createClient<paths>({ baseUrl, ...options });
}
